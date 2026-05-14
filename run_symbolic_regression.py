import torch
from typing import List, Tuple, Dict
from models import KAN, KANLinear
import os
from pysr import PySRRegressor
import warnings
import sympy

def extract_active_edges(layer: KANLinear, epsilon: float = 1e-3) -> List[Tuple[int, int]]:
    """
    Identifies unpruned edges in an external KAN layer based on parameter L1 norms.
    Returns a list of tuples (out_node, in_node).
    """
    # Compute L1 norm of the scaled spline weights over the spline coefficients
    # scaled_spline_weight shape: (out_features, in_features, coeff)
    edge_l1 = torch.sum(torch.abs(layer.scaled_spline_weight), dim=2)
    active_edges = torch.where(edge_l1 > epsilon)
    return list(zip(active_edges[0].tolist(), active_edges[1].tolist()))

def evaluate_edge(layer: KANLinear, in_idx: int, out_idx: int, x_in: torch.Tensor) -> torch.Tensor:
    """Evaluates a single 1D edge function from the efficient KAN."""
    batch_size = x_in.size(0)
    
    # The B-spline evaluation requires the full in_features dimension to match the grid shape
    X_dummy = torch.zeros(batch_size, layer.in_features, device=x_in.device)
    X_dummy[:, in_idx] = x_in
    
    # Evaluate base
    base_val = layer.base_activation(x_in) * layer.base_weight[out_idx, in_idx]
    
    # Evaluate spline
    bases = layer.b_splines(X_dummy) # (batch, in_features, coeff)
    bases_i = bases[:, in_idx, :] # (batch, coeff)
    spline_val = torch.sum(bases_i * layer.scaled_spline_weight[out_idx, in_idx], dim=1)
    
    return base_val + spline_val

def generate_pysr_dataset(model: KAN, num_samples: int = 200, epsilon: float = 1e-3) -> Dict[str, Dict]:
    device = next(model.parameters()).device
    
    # Generate 1D physiological points
    X = torch.empty(num_samples, model.layers[0].in_features).uniform_(0.1, 2.0).to(device)
    
    dataset = {'layer1': {}, 'layer2': {}}
    
    layer1 = model.layers[0]
    layer2 = model.layers[1]
    
    # --- Layer 1 Extraction ---
    active_l1 = extract_active_edges(layer1, epsilon)
    for (out_idx, in_idx) in active_l1:
        x_in = X[:, in_idx]
        with torch.no_grad():
            y_out = evaluate_edge(layer1, in_idx, out_idx, x_in)
        dataset['layer1'][(in_idx, out_idx)] = (x_in.cpu().numpy(), y_out.cpu().numpy())
        
    # --- Layer 2 Extraction ---
    with torch.no_grad():
        H = layer1(X)
        
    active_l2 = extract_active_edges(layer2, epsilon)
    for (out_idx, in_idx) in active_l2:
        h_in = H[:, in_idx]
        with torch.no_grad():
            y_out = evaluate_edge(layer2, in_idx, out_idx, h_in)
        dataset['layer2'][(in_idx, out_idx)] = (h_in.cpu().numpy(), y_out.cpu().numpy())
        
    return dataset


def fit_pysr_on_edges(dataset: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Fits symbolic expressions to each active 1D KAN edge using PySR.
    """
    equations = {'layer1': {}, 'layer2': {}}
    
    # Configure PySR for Biological/Dynamical ODEs
    pysr_model = PySRRegressor(
        niterations=20,
        timeout_in_seconds=5, # Limit each edge search to 5 seconds max
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["log", "exp", "neg"],
        complexity_of_operators={"log": 2, "exp": 2, "/": 2},
        parsimony=1e-3,  
        random_state=42,
        deterministic=False,
        parallelism="multithreading", 
        verbosity=0
    )
    

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The discovered expressions are being reset")
        
        l1_edges = list(dataset['layer1'].items())
        print(f"Fitting {len(l1_edges)} active edges in Layer 1...")
        for i, ((in_idx, out_idx), (x, y)) in enumerate(l1_edges, 1):
            print(f"  [{i}/{len(l1_edges)}] Fitting edge {in_idx}->{out_idx}...")
            # PySR requires 2D input arrays for features
            pysr_model.fit(x.reshape(-1, 1), y)
            equations['layer1'][(in_idx, out_idx)] = pysr_model.sympy()
            
        l2_edges = list(dataset['layer2'].items())
        print(f"\nFitting {len(l2_edges)} active edges in Layer 2...")
        for i, ((in_idx, out_idx), (x, y)) in enumerate(l2_edges, 1):
            print(f"  [{i}/{len(l2_edges)}] Fitting edge {in_idx}->{out_idx}...")
            pysr_model.fit(x.reshape(-1, 1), y)
            equations['layer2'][(in_idx, out_idx)] = pysr_model.sympy()
        
    return equations


def assemble_and_verify(equations: Dict[str, Dict], num_outputs: int = 3) -> Dict[int, sympy.Expr]:
    """
    Composes the 1D PySR equations into the macroscopic ODEs and scans for the Gompertzian term.
    """
    # Define physiological input variables
    p, q, r = sympy.symbols('p q r')
    input_vars = {0: p, 1: q, 2: r}
    
    # 1. Reconstruct Hidden Nodes: H_j = sum(Phi_{j, i}(x_i))
    hidden_nodes = {}
    for (in_idx, out_idx), eq in equations['layer1'].items():
        # PySR defaults the 1D input variable name to 'x0'
        x0 = sympy.Symbol('x0')
        subbed_eq = eq.subs(x0, input_vars[in_idx])
        
        hidden_nodes[out_idx] = hidden_nodes.get(out_idx, 0) + subbed_eq
        
    # 2. Reconstruct Output Derivatives: dX_k/dt = sum(Phi_{k, j}(H_j))
    macroscopic_odes = {k: sympy.Integer(0) for k in range(num_outputs)}
    for (in_idx, out_idx), eq in equations['layer2'].items():
        if in_idx not in hidden_nodes:
            continue  # Pruned hidden node
            
        x0 = sympy.Symbol('x0')
        subbed_eq = eq.subs(x0, hidden_nodes[in_idx])
        macroscopic_odes[out_idx] += subbed_eq
        
    # 3. Verification & Gompertzian Check
    print("\n" + "="*40)
    print("   RECONSTRUCTED MACROSCOPIC ODEs")
    print("="*40)
    
    found_gompertzian = False
    state_names = {0: "p", 1: "q", 2: "r"}
    
    for k in range(num_outputs):
        expr = macroscopic_odes[k]
        
        # OPTIMIZATION: Clean up computational noise BEFORE expanding.
        # This aggressively prunes the AST, preventing sympy.expand from stalling.
        replacements = {}
        for a in expr.atoms(sympy.Float):
            if abs(a) < 1e-4:
                replacements[a] = sympy.Integer(0)
            else:
                replacements[a] = sympy.Float(round(a, 4))
                
        cleaned_expr = expr.xreplace(replacements)
        
        cleaned_expr = sympy.expand(cleaned_expr)
        
        expr_str = str(cleaned_expr).replace(" ", "")
        
        print(f"\nd{state_names[k]}/dt = {cleaned_expr}")
        
        # Heuristic check for Gompertzian Tumor Growth: p * ln(p/q)
        # Accounts for expanded log identities
        is_gompertzian = any([
            ("p*log(p)" in expr_str and "p*log(q)" in expr_str),
            ("log(p/q)" in expr_str),
            ("p*log(p/q)" in expr_str)
        ])
        
        if is_gompertzian:
            print(f"  [SUCCESS] Gompertzian Growth Term identified in d{state_names[k]}/dt!")
            found_gompertzian = True
            
    if not found_gompertzian:
        print("\n[WARNING] Gompertzian term was not explicitly found.")
        
    return macroscopic_odes


def main():
    model_path = "kan_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run train.py first!")
        return

    print(f"Loading trained KAN model from {model_path}...")
    kan = KAN(layers_hidden=[3, 643, 3], grid_size=8, grid_range=[-3.0, 3.0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    kan.load_state_dict(torch.load(model_path, map_location=device))
    kan.to(device)
    kan.eval()

    print("\n" + "="*40)
    print("   PHASE II: SYMBOLIC EXTRACTION")
    print("="*40)
    dataset = generate_pysr_dataset(kan)
    eq_dict = fit_pysr_on_edges(dataset)
    final_odes = assemble_and_verify(eq_dict)

if __name__ == "__main__":
    main()
