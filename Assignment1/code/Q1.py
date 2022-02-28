import matplotlib.pyplot as plt
import utilities.common as common
import utilities.PCA as pca

if __name__ == '__main__':

    # ############################ Questsion 1 -> (i) Part #######################################
    
    print("---------------------------- Q1 - part (i) Output------------------------")

    # Load the data from given dataset
    original_data = common.load_data("dataset/Dataset.csv")
    centered_data = common.center_data(original_data)  # For centered data
    centered_data_variance = centered_data.var(
        axis=0)          # variance along x and y

    # Apply  PCA
    eigen_values, eigen_vectors = pca.fit(centered_data)

    # project data point in eigen space
    centered_projected_data = original_data @ eigen_vectors
    centered_projected_variance = centered_projected_data.var(
        axis=0)  # variance in data in eigen space

    # Total of variance along eigen basis will remain same as total of variance along original basis
    print("The variance along x and y axis in original centered data : ",
          centered_data_variance/centered_data_variance.sum())
    print("The variance along eigen axis ,in projected data : ",
          centered_projected_variance/centered_projected_variance.sum())
    print("Total variance of orginal and projected data",
          centered_data_variance.sum(), centered_projected_variance.sum())


# ################################# Questsion 1 -> (ii) Part ##################################
    # Not centered data case
    print("---------------------------- Q1 - part (ii) Output------------------------\n")
    # variance along x and y
    # without centring the data
    data_variance = original_data.var(axis=0)
    eigen_values2, eigen_vectors2 = pca.fit(original_data)  # PCA
    # project data point in eigen space
    projected_data = original_data @ eigen_vectors
    # variance in data in eigen space
    projected_variance = projected_data.var(axis=0)

    # Total of variance along eigen basis will remain same as total of variance along original basis
    print("The variance along x and y axis in original centered data : ",
          data_variance/data_variance.sum())
    print("The variance along eigen axis ,in projected data : ",
          projected_variance/projected_variance.sum())
    print("Total variance of orginal and projected data",
          data_variance.sum(), projected_variance.sum())

    common.plot_data(original_data, projected_data, eigen_values2, 'ro')
    plt.savefig('plots\Q1\A.png', dpi=300)

    common.plot_data(centered_data, centered_projected_data,eigen_values, 'bo')
    plt.savefig('plots\Q1\cA.png', dpi=300)

# #################################### Questsion 1 -> (iii) Part ###############################

    print("---------------------------- Q1 - part (iii) Output------------------------\n")

    for i in range(2, 4):
        kernal_eigen_values, kernal_projected_data = pca.kernalPCA(centered_data, pca.polynominal_function, i, no_ocmponents=2)

        common.plot_data(centered_data, kernal_projected_data, kernal_eigen_values, 'yo')
        keranl_projected_variance = kernal_projected_data.var(axis=0)

        print("Vaiance along eigen axis in projected data", keranl_projected_variance /
              keranl_projected_variance.sum(), '\n', "Total variance of projected data", keranl_projected_variance.sum())
       
        plt.savefig("plots\Q1\\"+str(i)+".png", dpi=500)

    for i in range(1, 11):
        kernal_eigen_values, kernal_projected_data = pca.kernalPCA(centered_data, pca.gauusian_function, i/10, no_ocmponents=2)

        common.plot_data(centered_data, kernal_projected_data, kernal_eigen_values, 'yo')
        keranl_projected_variance = kernal_projected_data.var(axis=0)

        print("Vaiance along eigen axis in projected data", keranl_projected_variance /
              keranl_projected_variance.sum(), '\n', "Total variance of projected data", keranl_projected_variance.sum())
   
        plt.savefig("plots\Q1\\"+str(i)+'k2.png', dpi=500)

print("---------------------------- Processing completed ------------------------")