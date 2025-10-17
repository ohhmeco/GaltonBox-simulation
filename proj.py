import random, math
import matplotlib.pyplot as plt

class TriangleBox:
    def __init__(self, trianglebase):
        self._base = trianglebase
        self._coordinates = [];
    def getBase(self):
        return self._base
    def draw(self):
        for i in range(0, self._base):
            print("\n")
            for j in range(0, i):
                print([i, j])
                self._coordinates.append([i, j]) 
    def getPoints(self):
        return self._coordinates

class GaltonBox:
    def __init__(self, matrix : TriangleBox, nballs : int):
        self._box = matrix
        self._nballs = nballs
        self._ballspositions = [];
        self._base = matrix._base
        self._bins = self._base + 1 #columns reachable in the experiment
        self._reachedBins = {}
        self._binsFrequencies = {}
        self._expectedBinomialFrequencies = {}
        self._expectedNormalFrequencies = {}
        self._errorPerBin = {}
        self._errorPerBinNormale = {}
        self._MSE = 0
        self._MSE_norm = 0

    def execute(self):
        self.simulate()
        self.analyzeFinalPositions()
        self.calculateBinsFrequencies()
        self.expectedFrequencies()
        self.confrontTheoryAndEmpiricResults()
        self.plotResults()

    def simulate(self):
        #Per ciascuna palla, per ciascun livello simulo 
        #se va a sinistra o se va a destra aggiornando ogni volta 
        #dove si trova.
        #print("Final level is " + str(self._base))
        for ball in range(self._nballs):

            position = [0, 0]
            #print(self._base)
            level = 0 
            while level < self._base:
                prob = random.random()
                if (prob > 1/2):
                    position[1] = position[1]+1 
                else:
                   position[0] = position[0]+1
                level+=1
            self.addFinalBallPosition(position)
            #print("New position is " + str(position))
    
    def addFinalBallPosition(self, pox : list):
        self._ballspositions.append(pox);


    def printFinalPositions(self):
        print("Final Positions reached: " + str(self._ballspositions))

    def analyzeFinalPositions(self):
        for x in range(self._bins):
            self._reachedBins[x] = 0;

        for x in self._ballspositions:
            self._reachedBins[x[0]]+=1

        print(self._reachedBins)

    def calculateBinsFrequencies(self):
        print(self._reachedBins.items())
        for index, x in self._reachedBins.items():
            self._binsFrequencies[index] = x  # x è già il conteggio (frequenza empirica)

    def expectedFrequencies(self):
        for index, x in self._reachedBins.items():
            #Bin
            prob_binomiale = math.comb(self._base, index) * (0.5**self._base) 
            self._expectedBinomialFrequencies[index] = prob_binomiale * self._nballs 
            
            # Norm 
            mu = self._base * 0.5
            variance = self._base *0.5 * 0.5
            devstandard = math.sqrt(variance)
            
            exponent = -((index - mu)**2) / (2 * variance) 
            normalizer = 1 / math.sqrt(variance * math.tau)
            
            pdf_value = normalizer * math.exp(exponent)
            self._expectedNormalFrequencies[index] = pdf_value * self._nballs
        
    def confrontTheoryAndEmpiricResults(self):

        for x in range(self._bins):
            # BIN
            errore_bin = self._binsFrequencies[x] - self._expectedBinomialFrequencies[x]
            self._errorPerBin[x] = pow(errore_bin, 2)
            self._MSE += self._errorPerBin[x]
            
            # NOR
            errore_norm = self._binsFrequencies[x] - self._expectedNormalFrequencies[x]
            self._errorPerBinNormale[x] = pow(errore_norm, 2)
            self._MSE_norm += self._errorPerBinNormale[x]

        self._MSE_norm /= self._bins
        self._MSE /= self._bins
        print(self._MSE)
        print(self._MSE_norm)

    def plotResults(self):
        #extract 
        bin_indices = list(range(self._bins))
        conteggi_empirici = [self._binsFrequencies[i] for i in bin_indices]
        bin_attesi = [self._expectedBinomialFrequencies[i] for i in bin_indices]
        norm_attesi = [self._expectedNormalFrequencies[i] for i in bin_indices]
        
        plt.figure(figsize=(12, 6)) # imposta le dimensioni del grafico
        plt.bar(bin_indices, conteggi_empirici, width=0.9, alpha=0.6, label=f'frequenza empirica ({self._nballs} palline)', color='skyblue')

        #bin
        plt.plot(bin_indices, bin_attesi, 'o--', color='darkblue', linewidth=1, markersize=5, label='teoria binomiale (esatta)')
        
        #norm
        plt.plot(bin_indices, norm_attesi, '-', color='red', linewidth=2, label='teoria normale (approssimazione)')
        
        plt.title(f'empirical vs. theoretical comparison in galton board (n={self._base}, n={self._nballs})')
        plt.xlabel('bin index (number of right/left turns)')
        plt.ylabel('count / expected frequency')
        
        mse_text = (f'mse empirical vs binomial: {self._MSE:.6f}\n'
                    f'mse empirical vs normal: {self._MSE_norm:.6f}')
        
        plt.text(0.05, 0.95, mse_text, 
                 transform=plt.gca().transAxes,  
                 fontsize=10, 
                 verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
        
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        #saving
        filename = f'galton_n{self._base}_balls{self._nballs}.png'
        plt.savefig(filename)
        plt.close() 

        print(f"\n[info] plot saved as {filename}")


def plotErrorComparison(results_data):
    
    if not results_data:
        print("[WARN] No error data to plot.")
        return

    labels = [f"n={d['n']}, N={d['N']}" for d in results_data]
    mse_binomial = [d['MSE_Binomial'] for d in results_data]
    mse_normal = [d['MSE_Normal'] for d in results_data]
    
    x = range(len(labels))  # Posizioni sull'asse X
    width = 0.35            # Larghezza delle barre
    
    plt.figure(figsize=(14, 7))
    
    # Plot dell'Errore Binomiale (MSE tra Empirico e Binomiale)
    rects1 = plt.bar([i - width/2 for i in x], mse_binomial, width, label='MSE vs. Binomial ', color='darkgreen', alpha=0.7)

    # Plot dell'Errore Normale (MSE tra Empirico e Normale)
    rects2 = plt.bar([i + width/2 for i in x], mse_normal, width, label='MSE vs. Normal ', color='orange', alpha=0.7)

    #format
    plt.title('Mean Squared Error (MSE) Comparison Across Different Experiments (n, N)')
    plt.xlabel('Experiment Configuration (n=Base, N=Balls)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()     

    plt.savefig('error_comparison_summary.png')
    plt.close()
    
    print("\n[INFO] Summary plot (error_comparison_summary.png) created.")



if __name__=="__main__":

    TEST_CONFIGS = [
        (10, 100),       # n medium, N small
        (10, 10000),     # n mediium, N big 
        (100, 1000),     # n big, N small
        (100, 100000),   # n big, N very big 
    ]
    
    results_data = [] 
    
    print("--- STARTING BATCH TEST CYCLE ---")
    
    for base, nballs in TEST_CONFIGS:
        
        print(f"\n========================================================")
        print(f"EXECUTING: Base n={base}, Balls N={nballs}")
        
        triangle_box = TriangleBox(base)
        galtonBox = GaltonBox(triangle_box, nballs)
        
        galtonBox.execute() #main function in class, calls all the others
        
        data = {
            'n': base,
            'N': nballs,
            'MSE_Binomial': galtonBox._MSE,
            'MSE_Normal': galtonBox._MSE_norm
        }
        results_data.append(data)
        
        print(f"MSE Empirical vs Binomial: {galtonBox._MSE:.6f}")
        print(f"MSE Empirical vs Normal: {galtonBox._MSE_norm:.6f}")
        
    plotErrorComparison(results_data)       



