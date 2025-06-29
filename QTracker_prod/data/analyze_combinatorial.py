import ROOT
import numpy as np
import argparse
import matplotlib.pyplot as plt

def analyze_combinatorial_data(input_file):
    """
    Analyze combinatorial data file to count mu+ and mu- particles and calculate their ratio.
    
    Args:
        input_file (str): Path to the ROOT file containing combinatorial data
    """
    
    # Open the ROOT file
    print(f"Opening file: {input_file}")
    f = ROOT.TFile.Open(input_file, "READ")
    
    if f.IsZombie():
        print(f"Error: Could not open file {input_file}")
        return
    
    # Get the tree
    tree = f.Get("tree")
    if not tree:
        print("Error: Could not find 'tree' in the ROOT file")
        f.Close()
        return
    
    print(f"Found tree with {tree.GetEntries()} entries")
    
    # Check which processID branch name is used
    def get_processID_branch_name(tree):
        branches = [branch.GetName() for branch in tree.GetListOfBranches()]
        if "gProcessID" in branches:
            return "gProcessID"
        elif "processID" in branches:
            return "processID"
        else:
            print(f"Warning: Neither 'gProcessID' nor 'processID' branch found. Available branches: {branches}")
            return None
    
    processID_name = get_processID_branch_name(tree)
    print(f"Using processID branch: {processID_name}")
    
    # Initialize counters
    mu_plus_count = 0
    mu_minus_count = 0
    total_events = 0
    events_with_muons = 0
    
    # Lists to store charge values for analysis
    all_charges = []
    event_charges = []
    
    print("Analyzing events...")
    
    # Loop through all entries
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        total_events += 1
        
        # Check if gCharge branch exists and has data
        if hasattr(tree, 'gCharge') and len(tree.gCharge) > 0:
            events_with_muons += 1
            
            # Count muons in this event
            event_mu_plus = 0
            event_mu_minus = 0
            
            for charge in tree.gCharge:
                all_charges.append(charge)
                event_charges.append(charge)
                
                if charge == 1:
                    mu_plus_count += 1
                    event_mu_plus += 1
                elif charge == -1:
                    mu_minus_count += 1
                    event_mu_minus += 1
                else:
                    print(f"Warning: Found charge value {charge} (not +1 or -1) in event {i}")
            
            # Print first few events for verification
            if i < 5:
                print(f"Event {i}: mu+ = {event_mu_plus}, mu- = {event_mu_minus}")
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} events...")
    
    # Calculate statistics
    total_muons = mu_plus_count + mu_minus_count
    mu_plus_ratio = mu_plus_count / max(total_muons, 1)
    mu_minus_ratio = mu_minus_count / max(total_muons, 1)
    mu_plus_mu_minus_ratio = mu_plus_count / max(mu_minus_count, 1)
    
    # Print results
    print("\n" + "="*60)
    print("COMBINATORIAL DATA ANALYSIS RESULTS")
    print("="*60)
    print(f"Total events processed: {total_events}")
    print(f"Events with muons: {events_with_muons}")
    print(f"Total muons found: {total_muons}")
    print(f"  - Mu+ (charge +1): {mu_plus_count}")
    print(f"  - Mu- (charge -1): {mu_minus_count}")
    print(f"  - Other charges: {total_muons - mu_plus_count - mu_minus_count}")
    print()
    print(f"Mu+ fraction: {mu_plus_ratio:.4f} ({mu_plus_ratio*100:.2f}%)")
    print(f"Mu- fraction: {mu_minus_ratio:.4f} ({mu_minus_ratio*100:.2f}%)")
    print(f"Mu+/Mu- ratio: {mu_plus_mu_minus_ratio:.4f}")
    print("="*60)
    
    # Create histogram of charge distribution
    if all_charges:
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.subplot(1, 2, 1)
        plt.hist(all_charges, bins=[-1.5, -0.5, 0.5, 1.5], 
                edgecolor='black', align='mid', alpha=0.7)
        plt.xlabel("Charge")
        plt.ylabel("Count")
        plt.title("Charge Distribution")
        plt.xticks([-1, 1], ['μ⁻ (-1)', 'μ⁺ (+1)'])
        plt.grid(True, alpha=0.3)
        
        # Create pie chart
        plt.subplot(1, 2, 2)
        labels = [f'μ⁺ ({mu_plus_count})', f'μ⁻ ({mu_minus_count})']
        sizes = [mu_plus_count, mu_minus_count]
        colors = ['lightcoral', 'lightblue']
        
        if total_muons > 0:
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title("Muon Charge Distribution")
        else:
            plt.text(0.5, 0.5, 'No muons found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("No Data")
        
        plt.tight_layout()
        plt.savefig("combinatorial_charge_analysis.png", dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: combinatorial_charge_analysis.png")
    
    # Close the file
    f.Close()
    
    return {
        'total_events': total_events,
        'events_with_muons': events_with_muons,
        'mu_plus_count': mu_plus_count,
        'mu_minus_count': mu_minus_count,
        'total_muons': total_muons,
        'mu_plus_ratio': mu_plus_ratio,
        'mu_minus_ratio': mu_minus_ratio,
        'mu_plus_mu_minus_ratio': mu_plus_mu_minus_ratio
    }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze combinatorial data file to count mu+ and mu- particles and calculate their ratio."
    )
    parser.add_argument("input_file", type=str, help="Path to the ROOT file containing combinatorial data")
    parser.add_argument("--output", type=str, default="combinatorial_analysis.txt",
                       help="Output text file for results (default: combinatorial_analysis.txt)")
    
    args = parser.parse_args()
    
    # Analyze the data
    results = analyze_combinatorial_data(args.input_file)
    
    # Save results to text file
    if results:
        with open(args.output, 'w') as f:
            f.write("COMBINATORIAL DATA ANALYSIS RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Input file: {args.input_file}\n")
            f.write(f"Total events processed: {results['total_events']}\n")
            f.write(f"Events with muons: {results['events_with_muons']}\n")
            f.write(f"Total muons found: {results['total_muons']}\n")
            f.write(f"  - Mu+ (charge +1): {results['mu_plus_count']}\n")
            f.write(f"  - Mu- (charge -1): {results['mu_minus_count']}\n")
            f.write(f"\nMu+ fraction: {results['mu_plus_ratio']:.4f} ({results['mu_plus_ratio']*100:.2f}%)\n")
            f.write(f"Mu- fraction: {results['mu_minus_ratio']:.4f} ({results['mu_minus_ratio']*100:.2f}%)\n")
            f.write(f"Mu+/Mu- ratio: {results['mu_plus_mu_minus_ratio']:.4f}\n")
        
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main() 