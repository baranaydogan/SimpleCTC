function [ctc sCT] = ContourTreeConnectivity(img,intensity_threshold,area_threshold)

% Checking and parsing inputs
if ((nargin~=1)&&(nargin~=3))
    error('Please check input arguments');
end

if (islogical(img) == 0)
    error('Input must be logical');
end

if (nargin == 1)
    intensity_threshold = 0;
    area_threshold      = 0;
end

if ((isscalar(intensity_threshold)&&isscalar(area_threshold))==0)
    error('Pruning parameters must be scalar');
end

% Computing signed Euclidean transform
boundary        = xor((bwdist(~img) > 1), img);
insideObject    = and((bwdist(~img) > 1), img);
outsideObject   = ~or(zeros(size(img)), or(boundary,insideObject));
edt             = double(bwdist(~insideObject) - bwdist(~outsideObject));

% Computing signed Euclidean transform
ct              = ContourTree(edt,intensity_threshold,area_threshold);

% Computing supplemented contour tree
[C,~,ic]        = unique(ct);
newVertices     = 1:length(C);

tree            = [edt(ct(:,1)) reshape(newVertices(ic), size(ct))];

sCT             = [];
newNode         = max(max(tree(:,2:3))) + 1;

for i = 1:size(tree,1)
    
    startLevel = round(tree(i,1));
    endLevel   = round(tree(find(tree(:,2) == tree(i,3),1),1));
    
    if ((startLevel-endLevel) > 1)
        
        sCT = [sCT ; [startLevel tree(i,2) newNode] ];
        
        for newLevel = (startLevel-1):-1:(endLevel+2)
            newNode  = newNode + 1;
            sCT      = [sCT ; [newLevel newNode-1 newNode] ];
        end
        
        sCT          = [sCT ; [endLevel+1 newNode tree(i,3)] ];
        newNode      = newNode + 1;
        
    elseif ((startLevel-endLevel) < -1)
        
        sCT = [sCT ; [startLevel tree(i,2) newNode] ];
        
        for newLevel = (startLevel+1):1:(endLevel-2)
            newNode  = newNode + 1;
            sCT      = [sCT ; [newLevel newNode-1 newNode] ];
        end
        
        sCT          = [sCT ; [endLevel-1 newNode tree(i,3)] ];
        newNode      = newNode + 1;
        
    else
        sCT = [sCT ; tree(i,:) ];
    end
    
end

[tmp sortIndices]   = sort(sCT(:,1), 1, 'descend'); %#ok<ASGLU>

sCT(:,1)            = sCT(sortIndices,1);
sCT(:,2)            = sCT(sortIndices,2);
sCT(:,3)            = sCT(sortIndices,3);

% Computing Laplacian matrix of supplementing contour tree
numberOfNodes                        = max(max(sCT(:, 2:3)));
contourTreeAdjacencyMatrix           = zeros( numberOfNodes );

for i = 1 : size(sCT,1)
    contourTreeAdjacencyMatrix(sCT(i,2), sCT(i,3)) = 1;
    contourTreeAdjacencyMatrix(i,i) = 0;
end

adjacencyMatrix             = tril(contourTreeAdjacencyMatrix) + triu(contourTreeAdjacencyMatrix)';
adjacencyMatrix             = adjacencyMatrix + tril(adjacencyMatrix)';

laplacianMatrix             = diag(sum(adjacencyMatrix)) - adjacencyMatrix;
[eigenVectors eigenValues]  = eig(laplacianMatrix, 'nobalance'); %#ok<ASGLU>

% Computing algebraic connectivity of supplemented contour tree
sortedEigenValues           = sort(diag(eigenValues));
algCon                      = sortedEigenValues(2);

% Computing CTC
levels                      = round(sCT(1,1)) - round(sCT(end,1)) + 1;
normFact                    = 2*(1-cos(pi/(levels)));
ctc                         = algCon / normFact;
