% Find the initial means, covariance matrices and mixing coefficients. 'data' is the original data matrix of size 13 by 128.

initMeans = kmeans; % initMeans are the initial means for GMM, where kmeans are the K means returned by K-means clustering implemented in Problem 2
initCovariances = zeros(dimension,numClusters); %initCovariances are the initial covariance matrices for GMM
initPriors = zeros(1,numClusters); %initPriors are initla mixing coefficients for GMM


for k=1:numClusters
    %data_k is the data matrix of size (#features by #points) containing points in the k-th cluster produced by K-means clustering implemented in Problem 2
    
    initPriors(i) = size(data_k,2) / size(data,2);

    if size(data_k,1) == 0 || size(data_k,2) == 0
        initCovariances(:,i) = diag(cov(data'));
    else
        initCovariances(:,i) = diag(cov(data_k'));
    end
end