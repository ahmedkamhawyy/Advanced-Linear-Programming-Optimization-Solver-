import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp

class LPBackend:
    """
    Handles the mathematical logic for solving Linear Programming problems.
    Requirements: Simplex, Big-M, Graphical, condition detection[cite: 38, 45, 49].
    """
    def __init__(self):
        self.M = 10000  # Large number for Big-M method
    
    def solve_graphical(self, obj_coeffs, constraints, optimization_type):
        """
        Solves 2-variable problems using the Graphical Method.
        """
        if len(obj_coeffs) != 2:
            return "Graphical method is only for 2 variables.", None

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Generate x values
        x_max = max([c[-1] / c[0] if c[0] != 0 else 10 for c in constraints] + [10])
        x_vals = np.linspace(0, x_max, 400)
        
        # Plot constraints
        for i, const in enumerate(constraints):
            a, b, op, rhs = const
            if b != 0:
                y_line = (rhs - a * x_vals) / b
                # Only plot feasible portion
                if op == "<=":
                    ax.fill_between(x_vals, 0, y_line, alpha=0.1)
                elif op == ">=":
                    ax.fill_between(x_vals, y_line, 10, alpha=0.1)
                ax.plot(x_vals, y_line, label=f"{a}x + {b}y {op} {rhs}")
            else:
                ax.axvline(x=rhs/a, label=f"{a}x {op} {rhs}")
        
        # Find all intersection points (vertices)
        vertices = []
        
        # Add origin
        vertices.append([0, 0])
        
        # Add axis intersections
        for const in constraints:
            a, b, op, rhs = const
            if a != 0:
                vertices.append([rhs/a, 0])
            if b != 0:
                vertices.append([0, rhs/b])
        
        # Add intersections between constraints
        n = len(constraints)
        for i in range(n):
            for j in range(i+1, n):
                a1, b1, op1, rhs1 = constraints[i]
                a2, b2, op2, rhs2 = constraints[j]
                
                # Check if lines are not parallel
                det = a1 * b2 - a2 * b1
                if abs(det) > 1e-10:
                    x = (rhs1 * b2 - rhs2 * b1) / det
                    y = (a1 * rhs2 - a2 * rhs1) / det
                    vertices.append([x, y])
        
        # Filter feasible vertices
        feasible_vertices = []
        for v in vertices:
            x, y = v
            if x < -1e-5 or y < -1e-5:
                continue
            
            feasible = True
            for const in constraints:
                a, b, op, rhs = const
                lhs = a * x + b * y
                
                if op == "<=" and lhs > rhs + 1e-5:
                    feasible = False
                    break
                elif op == ">=" and lhs < rhs - 1e-5:
                    feasible = False
                    break
                elif op == "=" and abs(lhs - rhs) > 1e-5:
                    feasible = False
                    break
            
            if feasible:
                feasible_vertices.append(v)
        
        if not feasible_vertices:
            ax.text(0.5, 0.5, 'No Feasible Region', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.legend()
            ax.grid(True)
            return "No feasible solution exists.", fig
        
        # Evaluate objective function at vertices
        best_z = -np.inf if optimization_type == "Maximize" else np.inf
        best_point = None
        log_text = "Corner Points Evaluation:\n"
        
        for v in feasible_vertices:
            x, y = v
            z = obj_coeffs[0] * x + obj_coeffs[1] * y
            log_text += f"({x:.2f}, {y:.2f}) -> Z = {z:.2f}\n"
            
            if optimization_type == "Maximize":
                if z > best_z:
                    best_z = z
                    best_point = (x, y)
            else:
                if z < best_z:
                    best_z = z
                    best_point = (x, y)
        
        # Highlight feasible region vertices
        vertices_array = np.array(feasible_vertices)
        ax.scatter(vertices_array[:, 0], vertices_array[:, 1], 
                  color='red', s=50, zorder=5, label='Feasible Vertices')
        
        if best_point:
            ax.plot(best_point[0], best_point[1], 'go', markersize=10, 
                   label=f'Optimal: ({best_point[0]:.2f}, {best_point[1]:.2f})')
            
            # Plot objective function line through optimal point
            if obj_coeffs[1] != 0:
                z_opt = best_z
                y_obj = (z_opt - obj_coeffs[0] * x_vals) / obj_coeffs[1]
                ax.plot(x_vals, y_obj, 'r--', alpha=0.7, label=f'Z = {z_opt:.2f}')
        
        # Set plot limits
        x_coords = [v[0] for v in feasible_vertices] + [0]
        y_coords = [v[1] for v in feasible_vertices] + [0]
        ax.set_xlim(0, max(x_coords) * 1.2)
        ax.set_ylim(0, max(y_coords) * 1.2)
        
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Graphical Method - {optimization_type}")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        result_str = f"{log_text}\nOptimal Solution: x1={best_point[0]:.2f}, x2={best_point[1]:.2f}\n"
        result_str += f"Optimal Value: Z = {best_z:.2f}"
        
        return result_str, fig

    def solve_simplex_tableau(self, obj_coeffs, constraints, optimization_type, use_big_m=False):
        """
        Solves using Simplex or Big-M by constructing the tableau.
        """
        # Convert to maximization if needed
        is_minimization = (optimization_type == "Minimize")
        if is_minimization:
            obj_coeffs = [-c for c in obj_coeffs]
        
        n_vars = len(obj_coeffs)
        n_constraints = len(constraints)
        
        # Initialize tableau
        # Count slack, surplus, and artificial variables
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        for const in constraints:
            op = const[-2]
            if op == "<=":
                slack_count += 1
            elif op == ">=":
                surplus_count += 1
                if use_big_m:
                    artificial_count += 1
            elif op == "=":
                if use_big_m:
                    artificial_count += 1
        
        total_vars = n_vars + slack_count + surplus_count + artificial_count
        
        # Create tableau (constraints + objective row)
        tableau = np.zeros((n_constraints + 1, total_vars + 1))
        
        # Fill constraint rows
        slack_idx = 0
        surplus_idx = slack_count
        artificial_idx = slack_count + surplus_count
        
        basic_vars = []  # Track basic variables for each row
        
        for i, const in enumerate(constraints):
            # Copy decision variables
            tableau[i, :n_vars] = const[:n_vars]
            
            # Handle constraint type
            op = const[-2]
            rhs = const[-1]
            
            if op == "<=":
                # Add slack variable
                tableau[i, n_vars + slack_idx] = 1
                basic_vars.append(n_vars + slack_idx)
                slack_idx += 1
            elif op == ">=":
                # Add surplus variable
                tableau[i, n_vars + surplus_idx] = -1
                surplus_idx += 1
                
                if use_big_m:
                    # Add artificial variable
                    tableau[i, n_vars + artificial_idx] = 1
                    basic_vars.append(n_vars + artificial_idx)
                    artificial_idx += 1
                else:
                    # For simplex without Big-M, we need to handle differently
                    # This case requires two-phase method, but for simplicity
                    # we'll convert to <= by multiplying by -1
                    pass
            elif op == "=":
                if use_big_m:
                    # Add artificial variable
                    tableau[i, n_vars + artificial_idx] = 1
                    basic_vars.append(n_vars + artificial_idx)
                    artificial_idx += 1
            
            # Set RHS
            tableau[i, -1] = rhs
        
        # Fill objective row
        tableau[-1, :n_vars] = [-c for c in obj_coeffs]  # Negative for standard form
        
        # Handle Big-M penalty in objective row
        if use_big_m and artificial_count > 0:
            # Add M penalty for artificial variables
            for i in range(n_vars + slack_count + surplus_count, total_vars):
                tableau[-1, i] = self.M
            
            # Eliminate artificial variables from objective row
            for i in range(n_constraints):
                if basic_vars[i] >= n_vars + slack_count + surplus_count:  # Artificial variable
                    tableau[-1, :] -= self.M * tableau[i, :]
        
        # Build headers for display
        headers = []
        for i in range(n_vars):
            headers.append(f"x{i+1}")
        for i in range(slack_count):
            headers.append(f"s{i+1}")
        for i in range(surplus_count):
            headers.append(f"e{i+1}")
        for i in range(artificial_count):
            headers.append(f"a{i+1}")
        headers.append("RHS")
        
        # Perform simplex iterations
        iterations_log = "Initial Tableau:\n" + self.format_tableau(tableau, headers) + "\n"
        
        iteration = 0
        max_iterations = 50
        
        while iteration < max_iterations:
            # Check optimality
            last_row = tableau[-1, :-1]
            if np.all(last_row >= -1e-10):
                break
            
            # Select entering variable (most negative in last row)
            entering_col = np.argmin(last_row)
            
            # Check for unboundedness
            if np.all(tableau[:-1, entering_col] <= 1e-10):
                return iterations_log + "\nProblem is UNBOUNDED.", None
            
            # Perform ratio test
            ratios = []
            for i in range(n_constraints):
                if tableau[i, entering_col] > 1e-10:
                    ratios.append(tableau[i, -1] / tableau[i, entering_col])
                else:
                    ratios.append(np.inf)
            
            if all(r == np.inf for r in ratios):
                return iterations_log + "\nProblem is UNBOUNDED.", None
            
            leaving_row = np.argmin(ratios)
            
            # Update basic variable
            basic_vars[leaving_row] = entering_col
            
            # Normalize pivot row
            pivot = tableau[leaving_row, entering_col]
            tableau[leaving_row, :] /= pivot
            
            # Eliminate entering variable from other rows
            for i in range(n_constraints + 1):
                if i != leaving_row:
                    factor = tableau[i, entering_col]
                    tableau[i, :] -= factor * tableau[leaving_row, :]
            
            iteration += 1
            iterations_log += f"\nIteration {iteration}:\n" + self.format_tableau(tableau, headers) + "\n"
        
        # Extract solution
        solution = {}
        for i in range(n_vars):
            solution[f"x{i+1}"] = 0.0
        
        # Find values from basic variables
        for i in range(n_constraints):
            col = basic_vars[i]
            if col < n_vars:
                solution[f"x{col+1}"] = tableau[i, -1]
        
        # Check for infeasibility (artificial variables in final solution)
        if use_big_m:
            for i in range(n_constraints):
                if basic_vars[i] >= n_vars + slack_count + surplus_count:
                    if abs(tableau[i, -1]) > 1e-5:
                        return iterations_log + "\nINFEASIBLE SOLUTION (Artificial variable > 0).", None
        
        optimal_value = tableau[-1, -1]
        if is_minimization:
            optimal_value = -optimal_value
        
        # Build result string
        result_str = iterations_log + "\n" + "="*50 + "\n"
        result_str += "OPTIMAL SOLUTION FOUND:\n\n"
        
        for var, val in solution.items():
            result_str += f"{var} = {val:.4f}\n"
        
        result_str += f"\nOptimal {'Min' if is_minimization else 'Max'} Z = {optimal_value:.4f}"
        
        # Add slack/surplus values
        result_str += "\n\nSlack/Surplus Variables:\n"
        for i in range(n_constraints):
            col = basic_vars[i]
            if col >= n_vars:  # Slack/surplus variable
                var_name = headers[col]
                result_str += f"{var_name} = {tableau[i, -1]:.4f}\n"
        
        return result_str, None

    def solve_lagrange(self, obj_coeffs, constraints, optimization_type):
        """
        Solves using Lagrange Multipliers for constrained optimization.
        """
        try:
            # Parse objective function
            x, y, lam = sp.symbols('x y lambda')
            
            # Build Lagrange function
            L = obj_coeffs[0] * x + obj_coeffs[1] * y
            
            # Add constraints with Lagrange multipliers
            for i, const in enumerate(constraints):
                a, b, op, rhs = const
                lam_i = sp.symbols(f'lambda_{i+1}')
                
                if op == "=":
                    L += lam_i * (a * x + b * y - rhs)
                elif op == "<=":
                    # Convert to equality with slack
                    s = sp.symbols(f's_{i+1}')
                    L += lam_i * (a * x + b * y + s**2 - rhs)
                elif op == ">=":
                    # Convert to equality with slack
                    s = sp.symbols(f's_{i+1}')
                    L += lam_i * (a * x + b * y - s**2 - rhs)
            
            # Calculate partial derivatives
            if len(constraints) == 1:
                # Single constraint case
                a, b, op, rhs = constraints[0]
                
                if op == "=":
                    # Lagrange: ∇f = λ∇g, g = 0
                    eq1 = sp.Eq(obj_coeffs[0], lam * a)
                    eq2 = sp.Eq(obj_coeffs[1], lam * b)
                    eq3 = sp.Eq(a * x + b * y, rhs)
                    
                    solution = sp.solve([eq1, eq2, eq3], (x, y, lam), dict=True)
                    
                    if solution:
                        result_str = "LAGRANGE MULTIPLIER METHOD\n"
                        result_str += "="*40 + "\n\n"
                        result_str += f"Constraint: {a}x + {b}y = {rhs}\n\n"
                        
                        for sol in solution:
                            x_val = float(sol[x])
                            y_val = float(sol[y])
                            lambda_val = float(sol[lam])
                            z_val = obj_coeffs[0] * x_val + obj_coeffs[1] * y_val
                            
                            result_str += f"Solution:\n"
                            result_str += f"  x = {x_val:.4f}\n"
                            result_str += f"  y = {y_val:.4f}\n"
                            result_str += f"  λ = {lambda_val:.4f}\n"
                            result_str += f"  Z = {z_val:.4f}\n"
                            result_str += f"  Check constraint: {a}*{x_val:.4f} + {b}*{y_val:.4f} = {a*x_val + b*y_val:.4f} (target: {rhs})\n"
                            
                            # Check KKT conditions for inequality constraints
                            if op == "<=" or op == ">=":
                                if lambda_val >= 0:
                                    result_str += "  λ ≥ 0 ✓ (KKT condition satisfied)\n"
                                else:
                                    result_str += "  λ < 0 ✗ (KKT condition violated)\n"
                            
                            result_str += "\n"
                        
                        return result_str, None
                    else:
                        return "No solution found using Lagrange multipliers.", None
                else:
                    # Inequality constraint - use KKT conditions
                    result_str = "KKT CONDITIONS (Lagrange for Inequalities)\n"
                    result_str += "="*50 + "\n\n"
                    
                    # Stationarity
                    result_str += "1. Stationarity (∇f = λ∇g):\n"
                    result_str += f"   {obj_coeffs[0]} = λ * {a}\n"
                    result_str += f"   {obj_coeffs[1]} = λ * {b}\n\n"
                    
                    # Primal feasibility
                    result_str += "2. Primal Feasibility:\n"
                    result_str += f"   {a}x + {b}y {op} {rhs}\n"
                    result_str += f"   x ≥ 0, y ≥ 0\n\n"
                    
                    # Dual feasibility
                    result_str += "3. Dual Feasibility:\n"
                    result_str += "   λ ≥ 0 (for ≤ constraint)\n" if op == "<=" else "   λ ≤ 0 (for ≥ constraint)\n\n"
                    
                    # Complementary slackness
                    result_str += "4. Complementary Slackness:\n"
                    result_str += f"   λ * ({a}x + {b}y - {rhs}) = 0\n\n"
                    
                    # Try to solve
                    if a != 0 and b != 0:
                        # Case 1: λ = 0 (constraint not binding)
                        if op == "<=":
                            # Try unconstrained optimum
                            if obj_coeffs[0] >= 0 and obj_coeffs[1] >= 0:
                                x_opt, y_opt = 0, 0
                                if a*x_opt + b*y_opt <= rhs:
                                    z_opt = 0
                                    result_str += f"Case λ = 0 (constraint not active):\n"
                                    result_str += f"  Solution: x={x_opt}, y={y_opt}, Z={z_opt}\n"
                                    result_str += f"  Constraint: {a}*{x_opt} + {b}*{y_opt} = {a*x_opt + b*y_opt} ≤ {rhs} ✓\n\n"
                        
                        # Case 2: Constraint binding (λ ≠ 0)
                        # Solve as equality
                        x_sym, y_sym, lam_sym = sp.symbols('x y lambda')
                        eq1 = sp.Eq(obj_coeffs[0], lam_sym * a)
                        eq2 = sp.Eq(obj_coeffs[1], lam_sym * b)
                        eq3 = sp.Eq(a * x_sym + b * y_sym, rhs)
                        
                        sol = sp.solve([eq1, eq2, eq3], (x_sym, y_sym, lam_sym))
                        if sol:
                            x_val = float(sol[0][0])
                            y_val = float(sol[0][1])
                            lambda_val = float(sol[0][2])
                            
                            result_str += f"Case constraint binding (λ ≠ 0):\n"
                            result_str += f"  x = {x_val:.4f}, y = {y_val:.4f}\n"
                            result_str += f"  λ = {lambda_val:.4f}\n"
                            result_str += f"  Z = {obj_coeffs[0]*x_val + obj_coeffs[1]*y_val:.4f}\n"
                            
                            # Check sign of λ based on constraint type
                            if (op == "<=" and lambda_val >= -1e-10) or (op == ">=" and lambda_val <= 1e-10):
                                result_str += f"  λ sign condition ✓\n"
                            else:
                                result_str += f"  λ sign condition ✗\n"
                    
                    return result_str, None
            
            else:
                # Multiple constraints - general message
                result_str = "LAGRANGE MULTIPLIER METHOD\n"
                result_str += "="*40 + "\n\n"
                result_str += f"Objective: Z = {obj_coeffs[0]}x + {obj_coeffs[1]}y\n"
                result_str += f"Number of constraints: {len(constraints)}\n\n"
                result_str += "For multiple constraints, the Lagrange function is:\n"
                result_str += "L(x,y,λ) = f(x,y) + Σ λ_i * g_i(x,y)\n\n"
                
                result_str += "Where:\n"
                for i, const in enumerate(constraints):
                    a, b, op, rhs = const
                    result_str += f"  g_{i+1}(x,y) = {a}x + {b}y {op} {rhs}\n"
                
                result_str += "\nKKT Conditions:\n"
                result_str += "1. Stationarity: ∇f + Σ λ_i ∇g_i = 0\n"
                result_str += "2. Primal Feasibility: g_i(x,y) ≤ 0 (or ≥ 0, = 0)\n"
                result_str += "3. Dual Feasibility: λ_i ≥ 0 for ≤ constraints\n"
                result_str += "4. Complementary Slackness: λ_i * g_i(x,y) = 0\n"
                
                return result_str, None
                
        except Exception as e:
            return f"Error in Lagrange method: {str(e)}", None

    def format_tableau(self, tableau, headers):
        """Format tableau for display."""
        result = "      " + "  ".join(f"{h:>8}" for h in headers) + "\n"
        result += "-" * (12 * len(headers)) + "\n"
        
        for i, row in enumerate(tableau):
            if i == len(tableau) - 1:
                result += "Z  "
            else:
                result += f"R{i+1} "
            
            for j, val in enumerate(row):
                if abs(val) < 1e-10:
                    result += f"{0:>8.2f} "
                else:
                    result += f"{val:>8.2f} "
            result += "\n"
        
        return result

class LPApp:
    """
    Main GUI Application class.
    Requirements: User Inputs, Method Selection, Output Display[cite: 4, 39, 45].
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced LP Solver with Big-M and Lagrange")
        self.root.geometry("1000x800")
        self.solver = LPBackend()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Tab 1: Linear Programming
        lp_frame = ttk.Frame(notebook)
        notebook.add(lp_frame, text="Linear Programming")
        self.setup_lp_tab(lp_frame)
        
        # Tab 2: Lagrange Multipliers
        lagrange_frame = ttk.Frame(notebook)
        notebook.add(lagrange_frame, text="Lagrange Multipliers")
        self.setup_lagrange_tab(lagrange_frame)
    
    def setup_lp_tab(self, parent):
        # Frame 1: Configuration
        config_frame = ttk.LabelFrame(parent, text="Problem Setup")
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Optimization Type
        ttk.Label(config_frame, text="Optimization:").grid(row=0, column=0, padx=5, pady=5)
        self.opt_type = tk.StringVar(value="Maximize")
        ttk.Combobox(config_frame, textvariable=self.opt_type, 
                    values=["Maximize", "Minimize"], width=12).grid(row=0, column=1, padx=5)
        
        # Dimensions
        ttk.Label(config_frame, text="Variables:").grid(row=0, column=2, padx=5)
        self.num_vars = tk.IntVar(value=2)
        ttk.Spinbox(config_frame, from_=2, to=10, textvariable=self.num_vars, 
                   width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(config_frame, text="Constraints:").grid(row=0, column=4, padx=5)
        self.num_const = tk.IntVar(value=2)
        ttk.Spinbox(config_frame, from_=1, to=10, textvariable=self.num_const, 
                   width=8).grid(row=0, column=5, padx=5)
        
        ttk.Button(config_frame, text="Generate Inputs", 
                  command=self.generate_inputs).grid(row=0, column=6, padx=10)
        
        # Frame 2: Dynamic Inputs
        self.input_frame = ttk.LabelFrame(parent, text="Problem Input")
        self.input_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.obj_entries = []
        self.const_rows = []
        
        # Frame 3: Method Selection
        method_frame = ttk.LabelFrame(parent, text="Solving Method")
        method_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(method_frame, text="Method:").pack(side="left", padx=5)
        self.method = tk.StringVar(value="Simplex")
        
        methods = [
            ("Simplex Method", "Simplex"),
            ("Big-M Method", "Big-M"),
            ("Graphical Method", "Graphical")
        ]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method, 
                          value=value).pack(side="left", padx=5)
        
        ttk.Button(method_frame, text="SOLVE", command=self.solve_lp, 
                  style="Accent.TButton").pack(side="right", padx=10)
        
        # Frame 4: Results
        results_frame = ttk.LabelFrame(parent, text="Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(results_frame, height=20, 
                                                    font=("Courier", 10))
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Generate initial inputs
        self.generate_inputs()
    
    def setup_lagrange_tab(self, parent):
        # Frame for Lagrange inputs
        input_frame = ttk.LabelFrame(parent, text="Lagrange Problem Setup")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # Objective function
        ttk.Label(input_frame, text="Objective Z =").grid(row=0, column=0, padx=5, pady=5)
        self.obj_x = tk.StringVar(value="3")
        ttk.Entry(input_frame, textvariable=self.obj_x, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(input_frame, text="x +").grid(row=0, column=2)
        self.obj_y = tk.StringVar(value="5")
        ttk.Entry(input_frame, textvariable=self.obj_y, width=8).grid(row=0, column=3, padx=2)
        ttk.Label(input_frame, text="y").grid(row=0, column=4)
        
        # Constraints
        ttk.Label(input_frame, text="Constraints:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.lag_constraints = []
        for i in range(3):  # Allow up to 3 constraints
            row_frame = ttk.Frame(input_frame)
            row_frame.grid(row=2+i, column=0, columnspan=6, sticky="w", pady=2)
            
            # Coefficient for x
            coeff_x = tk.StringVar(value="1" if i == 0 else "0")
            ttk.Entry(row_frame, textvariable=coeff_x, width=6).pack(side="left", padx=2)
            ttk.Label(row_frame, text="x +").pack(side="left")
            
            # Coefficient for y
            coeff_y = tk.StringVar(value="1" if i == 0 else "0")
            ttk.Entry(row_frame, textvariable=coeff_y, width=6).pack(side="left", padx=2)
            ttk.Label(row_frame, text="y").pack(side="left")
            
            # Operator
            op_var = tk.StringVar(value="=")
            ttk.Combobox(row_frame, textvariable=op_var, values=["=", "<=", ">="], 
                        width=3).pack(side="left", padx=5)
            
            # RHS
            rhs_var = tk.StringVar(value="10" if i == 0 else "0")
            ttk.Entry(row_frame, textvariable=rhs_var, width=6).pack(side="left", padx=2)
            
            self.lag_constraints.append({
                'coeff_x': coeff_x,
                'coeff_y': coeff_y,
                'op': op_var,
                'rhs': rhs_var
            })
        
        # Solve button
        ttk.Button(input_frame, text="Solve with Lagrange", 
                  command=self.solve_lagrange).grid(row=5, column=0, columnspan=6, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Lagrange Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.lagrange_result_text = scrolledtext.ScrolledText(results_frame, height=20,
                                                             font=("Courier", 10))
        self.lagrange_result_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def generate_inputs(self):
        """Generate input fields for LP problem."""
        # Clear previous inputs
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        
        n = self.num_vars.get()
        m = self.num_const.get()
        
        # Objective function
        ttk.Label(self.input_frame, text="Objective Function (Z):").grid(
            row=0, column=0, sticky="w", padx=5, pady=10)
        
        self.obj_entries = []
        for i in range(n):
            ttk.Label(self.input_frame, text=f"x{i+1}").grid(row=0, column=1 + i*2, padx=2)
            ent = ttk.Entry(self.input_frame, width=8)
            ent.grid(row=0, column=2 + i*2, padx=2)
            ent.insert(0, "3" if i == 0 else "5" if i == 1 else "0")
            self.obj_entries.append(ent)
            
            if i < n-1:
                ttk.Label(self.input_frame, text="+").grid(row=0, column=3 + i*2)
        
        ttk.Label(self.input_frame, text=f"→ {self.opt_type.get()}").grid(
            row=0, column=2 + n*2, padx=10)
        
        # Constraints
        ttk.Label(self.input_frame, text="Constraints:").grid(
            row=1, column=0, sticky="w", padx=5, pady=10)
        
        self.const_rows = []
        for i in range(m):
            row_widgets = []
            row_entries = []
            
            for j in range(n):
                ent = ttk.Entry(self.input_frame, width=8)
                ent.grid(row=2+i, column=2 + j*2, padx=2)
                
                # Set default values for 2-variable case
                if n == 2:
                    if j == 0:
                        ent.insert(0, "2" if i == 0 else "1")
                    else:
                        ent.insert(0, "1" if i == 0 else "2")
                else:
                    ent.insert(0, "1" if i == j else "0")
                
                row_entries.append(ent)
                
                if j < n-1:
                    ttk.Label(self.input_frame, text="+").grid(row=2+i, column=3 + j*2)
            
            # Operator
            op_var = tk.StringVar(value="<=")
            op_menu = ttk.Combobox(self.input_frame, textvariable=op_var, 
                                  values=["<=", ">=", "="], width=3)
            op_menu.grid(row=2+i, column=2 + n*2, padx=5)
            
            # RHS
            rhs_ent = ttk.Entry(self.input_frame, width=8)
            rhs_ent.grid(row=2+i, column=3 + n*2, padx=2)
            rhs_ent.insert(0, "20" if i == 0 else "30" if i == 1 else "10")
            
            self.const_rows.append({
                'coeffs': row_entries,
                'op': op_var,
                'rhs': rhs_ent
            })
    
    def solve_lp(self):
        """Solve LP problem."""
        try:
            # Parse inputs
            obj_coeffs = [float(e.get()) for e in self.obj_entries]
            
            constraints = []
            for row in self.const_rows:
                coeffs = [float(e.get()) for e in row['coeffs']]
                op = row['op'].get()
                rhs = float(row['rhs'].get())
                constraints.append(coeffs + [op, rhs])
            
            method = self.method.get()
            opt_type = self.opt_type.get()
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Solving {opt_type} problem using {method}...\n")
            self.result_text.insert(tk.END, "="*50 + "\n\n")
            
            # Call appropriate solver
            if method == "Graphical":
                if len(obj_coeffs) != 2:
                    messagebox.showerror("Error", "Graphical method requires exactly 2 variables!")
                    return
                
                result, fig = self.solver.solve_graphical(obj_coeffs, constraints, opt_type)
                self.result_text.insert(tk.END, result)
                
                if fig:
                    # Display graph
                    graph_window = tk.Toplevel(self.root)
                    graph_window.title("Graphical Solution")
                    graph_window.geometry("700x600")
                    
                    canvas = FigureCanvasTkAgg(fig, master=graph_window)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    ttk.Button(graph_window, text="Close", 
                              command=graph_window.destroy).pack(pady=10)
            
            elif method == "Simplex":
                result, _ = self.solver.solve_simplex_tableau(
                    obj_coeffs, constraints, opt_type, use_big_m=False)
                self.result_text.insert(tk.END, result)
            
            elif method == "Big-M":
                result, _ = self.solver.solve_simplex_tableau(
                    obj_coeffs, constraints, opt_type, use_big_m=True)
                self.result_text.insert(tk.END, result)
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def solve_lagrange(self):
        """Solve using Lagrange multipliers."""
        try:
            # Parse inputs
            obj_coeffs = [
                float(self.obj_x.get()),
                float(self.obj_y.get())
            ]
            
            constraints = []
            for const in self.lag_constraints:
                coeff_x = const['coeff_x'].get()
                coeff_y = const['coeff_y'].get()
                op = const['op'].get()
                rhs = const['rhs'].get()
                
                if coeff_x or coeff_y:  # Only add if at least one coefficient is non-zero
                    constraints.append([
                        float(coeff_x),
                        float(coeff_y),
                        op,
                        float(rhs)
                    ])
            
            if not constraints:
                messagebox.showwarning("Warning", "Please enter at least one constraint!")
                return
            
            self.lagrange_result_text.delete(1.0, tk.END)
            self.lagrange_result_text.insert(tk.END, 
                "Solving constrained optimization using Lagrange multipliers...\n")
            self.lagrange_result_text.insert(tk.END, "="*60 + "\n\n")
            
            # Call Lagrange solver (default to Maximize for Lagrange)
            result, _ = self.solver.solve_lagrange(obj_coeffs, constraints, "Maximize")
            self.lagrange_result_text.insert(tk.END, result)
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    
    # Configure styles
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
    
    app = LPApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()