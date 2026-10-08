import ast
import os
from pathlib import Path
import argparse

def get_python_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                yield Path(root) / file

def fix_file(filepath, functions_to_fix, interactive=True):
    """
    functions_to_fix: list of ast.FunctionDef nodes
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    fixes_applied = 0
        
    # On trie en ordre inverse pour que la modification d'une ligne n'impacte pas les numéros de ligne des fonctions précédentes
    for node in sorted(functions_to_fix, key=lambda n: n.lineno, reverse=True):
        start_line = node.lineno - 1
        
        in_string = False
        string_char = None
        open_parens = 0
        found_def = False
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            modified = False
            j = 0
            while j < len(line):
                char = line[j]
                
                # Ignorer l'intérieur des chaînes de caractères
                if in_string:
                    if char == '\\':
                        j += 2
                        continue
                    elif char == string_char:
                        in_string = False
                elif char in ("'", '"'):
                    in_string = True
                    string_char = char
                # Ignorer les commentaires
                elif char == '#':
                    break
                else:
                    if not found_def:
                        if char == '(':
                            open_parens += 1
                            found_def = True
                    else:
                        if char == '(':
                            open_parens += 1
                        elif char == ')':
                            open_parens -= 1
                            if open_parens == 0:
                                # On a trouvé la parenthèse fermante de la signature
                                prefix = line[:j]
                                suffix = line[j:]
                                clean_prefix = prefix.strip()
                                
                                if clean_prefix.endswith('('):
                                    insert_str = "*args, **kwargs"
                                elif clean_prefix.endswith(','):
                                    insert_str = " *args, **kwargs"
                                else:
                                    insert_str = ", *args, **kwargs"
                                    
                                new_line = prefix + insert_str + suffix
                                
                                if interactive:
                                    print(f"\nFichier : {filepath.name} (Ligne {i+1})")
                                    print(f"[-] Avant : {line.strip()}")
                                    print(f"[+] Après : {new_line.strip()}")
                                    
                                    while True:
                                        resp = input("Appliquer cette correction ? [O/n] : ").strip().lower()
                                        if resp in ('', 'o', 'oui', 'y', 'yes'):
                                            lines[i] = new_line
                                            fixes_applied += 1
                                            modified = True
                                            break
                                        elif resp in ('n', 'non', 'no'):
                                            print("-> Correction ignorée.")
                                            modified = True
                                            break
                                else:
                                    lines[i] = new_line
                                    fixes_applied += 1
                                    modified = True
                                break
                j += 1
            if modified:
                break
                
    if fixes_applied > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  -> {fixes_applied} correction(s) sauvegardée(s) dans {filepath.name}")

def check_callbacks_in_file(filepath, auto_fix=False):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(filepath))
    except Exception as e:
        print(f"Failed to parse {filepath}: {e}")
        return

    function_signatures = {}
    
    class FunctionVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            has_vararg = node.args.vararg is not None
            has_kwarg = node.args.kwarg is not None
            pos_args = len(node.args.args)
            
            function_signatures[node.name] = {
                'has_vararg': has_vararg,
                'has_kwarg': has_kwarg,
                'pos_args': pos_args,
                'node': node
            }
            self.generic_visit(node)

    FunctionVisitor().visit(tree)

    class CallbackVisitor(ast.NodeVisitor):
        def __init__(self):
            self.issues_found = []
            self.nodes_to_fix = []

        def visit_Call(self, node):
            for kw in node.keywords:
                if kw.arg == "callback":
                    callback_name = None
                    if isinstance(kw.value, ast.Name):
                        callback_name = kw.value.id
                    elif isinstance(kw.value, ast.Attribute):
                        callback_name = kw.value.attr
                        
                    if callback_name and callback_name in function_signatures:
                        sig = function_signatures[callback_name]
                        if not sig['has_vararg'] and not sig['has_kwarg']:
                            if sig['node'] not in self.nodes_to_fix:
                                self.issues_found.append(
                                    f"Ligne {kw.value.lineno}: Le callback '{callback_name}' nécessiterait *args, **kwargs."
                                )
                                self.nodes_to_fix.append(sig['node'])
            self.generic_visit(node)

    visitor = CallbackVisitor()
    visitor.visit(tree)
    
    if visitor.issues_found and not auto_fix:
        print(f"\n--- Fichier : {filepath.name} ---")
        for issue in visitor.issues_found:
            print(issue)
            
    if auto_fix and visitor.nodes_to_fix:
        fix_file(filepath, visitor.nodes_to_fix, interactive=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse et corrige interactivement les signatures des callbacks DPG.")
    parser.add_argument('--fix', action='store_true', help="Démarre la correction interactive.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    
    if args.fix:
        print("Démarrage de l'assistant de correction interactif...")
    else:
        print("Analyse des callbacks DPG dans le projet...")
        
    dirs_to_check = [
        project_root / "core",
        project_root / "modules",
        project_root / "config",
    ]
    
    for directory in dirs_to_check:
        if directory.exists():
            for py_file in get_python_files(directory):
                check_callbacks_in_file(py_file, auto_fix=args.fix)
                
    if args.fix:
        print("\nSession de correction terminée !")
    else:
        print("\nAnalyse terminée. Pour corriger interactivement, utilisez : python tools/check_callbacks.py --fix")
