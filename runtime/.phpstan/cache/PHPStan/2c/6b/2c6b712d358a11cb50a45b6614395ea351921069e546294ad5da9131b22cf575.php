<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Clusterers/KMeans.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Clusterers\KMeans
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-93a2486446db39a7e7819a8275e5ecd34f896473bc328281b44fed452d7be24b',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Clusterers\\KMeans',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Clusterers/KMeans.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Clusterers',
    'name' => 'Rubix\\ML\\Clusterers\\KMeans',
    'shortName' => 'KMeans',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * K Means
 *
 * A fast online centroid-based hard clustering algorithm capable of grouping linearly
 * separable data points given some prior knowledge of the target number of clusters
 * (defined by *k*). K Means is trained using adaptive Mini Batch Gradient Descent and
 * minimizes the inertia cost function. Inertia is defined as the average sum of distances
 * between each sample and its nearest cluster centroid.
 *
 * References:
 * [1] D. Sculley. (2010). Web-Scale K-Means Clustering.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 55,
    'endLine' => 570,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\Online',
      3 => 'Rubix\\ML\\Probabilistic',
      4 => 'Rubix\\ML\\Verbose',
      5 => 'Rubix\\ML\\Persistable',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AutotrackRevisions',
      1 => 'Rubix\\ML\\Traits\\LoggerAware',
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'k' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'k',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The target number of clusters.
 *
 * @var int<0,max>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 64,
        'endLine' => 64,
        'startColumn' => 5,
        'endColumn' => 21,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'batchSize' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'batchSize',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The size of each mini batch in samples.
 *
 * @var positive-int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 71,
        'endLine' => 71,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'epochs' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'epochs',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The maximum number of iterations to run until the algorithm terminates.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 78,
        'endLine' => 78,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'minChange' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'minChange',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The minimum change in the inertia for training to continue.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 85,
        'endLine' => 85,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'window' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'window',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The number of epochs without improvement in the training loss to wait before considering an early stop.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 92,
        'endLine' => 92,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'kernel' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'kernel',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Kernels\\Distance\\Distance',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The distance function to use when computing the distances.
 *
 * @var Distance
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 99,
        'endLine' => 99,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'seeder' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'seeder',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Clusterers\\Seeders\\Seeder',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The cluster centroid seeder.
 *
 * @var Seeder
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 106,
        'endLine' => 106,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'centroids' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'centroids',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'default' => 
        array (
          'code' => '[]',
          'attributes' => 
          array (
            'startLine' => 113,
            'endLine' => 115,
            'startTokenPos' => 282,
            'startFilePos' => 2885,
            'endTokenPos' => 286,
            'endFilePos' => 2902,
          ),
        ),
        'docComment' => '/**
 * The computed centroid vectors of the training data.
 *
 * @var list<list<int|float>>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 113,
        'endLine' => 115,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'sizes' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'sizes',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'default' => 
        array (
          'code' => '[]',
          'attributes' => 
          array (
            'startLine' => 122,
            'endLine' => 124,
            'startTokenPos' => 299,
            'startFilePos' => 3054,
            'endTokenPos' => 303,
            'endFilePos' => 3071,
          ),
        ),
        'docComment' => '/**
 * The number of training samples contained within each cluster centroid.
 *
 * @var int[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 122,
        'endLine' => 124,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'losses' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'name' => 'losses',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
          'data' => 
          array (
            'types' => 
            array (
              0 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'array',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'null',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 131,
            'endLine' => 131,
            'startTokenPos' => 317,
            'startFilePos' => 3216,
            'endTokenPos' => 317,
            'endFilePos' => 3219,
          ),
        ),
        'docComment' => '/**
 * The loss at each epoch from the last training session.
 *
 * @var float[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 131,
        'endLine' => 131,
        'startColumn' => 5,
        'endColumn' => 36,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
    ),
    'immediateMethods' => 
    array (
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'k' => 
          array (
            'name' => 'k',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 144,
            'endLine' => 144,
            'startColumn' => 9,
            'endColumn' => 14,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'batchSize' => 
          array (
            'name' => 'batchSize',
            'default' => 
            array (
              'code' => '128',
              'attributes' => 
              array (
                'startLine' => 145,
                'endLine' => 145,
                'startTokenPos' => 340,
                'startFilePos' => 3556,
                'endTokenPos' => 340,
                'endFilePos' => 3558,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 145,
            'endLine' => 145,
            'startColumn' => 9,
            'endColumn' => 28,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'epochs' => 
          array (
            'name' => 'epochs',
            'default' => 
            array (
              'code' => '1000',
              'attributes' => 
              array (
                'startLine' => 146,
                'endLine' => 146,
                'startTokenPos' => 349,
                'startFilePos' => 3583,
                'endTokenPos' => 349,
                'endFilePos' => 3586,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 146,
            'endLine' => 146,
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
          'minChange' => 
          array (
            'name' => 'minChange',
            'default' => 
            array (
              'code' => '0.0001',
              'attributes' => 
              array (
                'startLine' => 147,
                'endLine' => 147,
                'startTokenPos' => 358,
                'startFilePos' => 3616,
                'endTokenPos' => 358,
                'endFilePos' => 3619,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 147,
            'endLine' => 147,
            'startColumn' => 9,
            'endColumn' => 31,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
          'window' => 
          array (
            'name' => 'window',
            'default' => 
            array (
              'code' => '5',
              'attributes' => 
              array (
                'startLine' => 148,
                'endLine' => 148,
                'startTokenPos' => 367,
                'startFilePos' => 3644,
                'endTokenPos' => 367,
                'endFilePos' => 3644,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 148,
            'endLine' => 148,
            'startColumn' => 9,
            'endColumn' => 23,
            'parameterIndex' => 4,
            'isOptional' => true,
          ),
          'kernel' => 
          array (
            'name' => 'kernel',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 149,
                'endLine' => 149,
                'startTokenPos' => 377,
                'startFilePos' => 3675,
                'endTokenPos' => 377,
                'endFilePos' => 3678,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
              'data' => 
              array (
                'types' => 
                array (
                  0 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'Rubix\\ML\\Kernels\\Distance\\Distance',
                      'isIdentifier' => false,
                    ),
                  ),
                  1 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'null',
                      'isIdentifier' => true,
                    ),
                  ),
                ),
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 149,
            'endLine' => 149,
            'startColumn' => 9,
            'endColumn' => 32,
            'parameterIndex' => 5,
            'isOptional' => true,
          ),
          'seeder' => 
          array (
            'name' => 'seeder',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 150,
                'endLine' => 150,
                'startTokenPos' => 387,
                'startFilePos' => 3707,
                'endTokenPos' => 387,
                'endFilePos' => 3710,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
              'data' => 
              array (
                'types' => 
                array (
                  0 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'Rubix\\ML\\Clusterers\\Seeders\\Seeder',
                      'isIdentifier' => false,
                    ),
                  ),
                  1 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'null',
                      'isIdentifier' => true,
                    ),
                  ),
                ),
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 150,
            'endLine' => 150,
            'startColumn' => 9,
            'endColumn' => 30,
            'parameterIndex' => 6,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $k
 * @param int $batchSize
 * @param int $epochs
 * @param float $minChange
 * @param int $window
 * @param Distance|null $kernel
 * @param Seeder|null $seeder
 * @throws InvalidArgumentException
 */',
        'startLine' => 143,
        'endLine' => 184,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'type' => 
      array (
        'name' => 'type',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\EstimatorType',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the estimator type.
 *
 * @return EstimatorType
 */',
        'startLine' => 191,
        'endLine' => 194,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'compatibility' => 
      array (
        'name' => 'compatibility',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the data types that the estimator is compatible with.
 *
 * @return list<DataType>
 */',
        'startLine' => 201,
        'endLine' => 206,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'params' => 
      array (
        'name' => 'params',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the settings of the hyper-parameters in an associative array.
 *
 * @return mixed[]
 */',
        'startLine' => 213,
        'endLine' => 224,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'trained' => 
      array (
        'name' => 'trained',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Has the learner been trained?
 *
 * @return bool
 */',
        'startLine' => 231,
        'endLine' => 234,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'centroids' => 
      array (
        'name' => 'centroids',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the computed cluster centroids of the training data.
 *
 * @return list<list<int|float>>
 */',
        'startLine' => 241,
        'endLine' => 244,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'sizes' => 
      array (
        'name' => 'sizes',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the number of training samples each centroid is responsible for.
 *
 * @return int[]
 */',
        'startLine' => 251,
        'endLine' => 254,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'steps' => 
      array (
        'name' => 'steps',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Generator',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return an iterable progress table with the steps from the last training session.
 *
 * @return Generator<mixed[]>
 */',
        'startLine' => 261,
        'endLine' => 273,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'losses' => 
      array (
        'name' => 'losses',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
          'data' => 
          array (
            'types' => 
            array (
              0 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'array',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'null',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the loss for each epoch from the last training session.
 *
 * @return float[]|null
 */',
        'startLine' => 280,
        'endLine' => 283,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'train' => 
      array (
        'name' => 'train',
        'parameters' => 
        array (
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Dataset',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 290,
            'endLine' => 290,
            'startColumn' => 27,
            'endColumn' => 42,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Train the learner with a dataset.
 *
 * @param Dataset $dataset
 */',
        'startLine' => 290,
        'endLine' => 308,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'partial' => 
      array (
        'name' => 'partial',
        'parameters' => 
        array (
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Dataset',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 315,
            'endLine' => 315,
            'startColumn' => 29,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Perform a partial train on the learner.
 *
 * @param Dataset $dataset
 */',
        'startLine' => 315,
        'endLine' => 432,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'predict' => 
      array (
        'name' => 'predict',
        'parameters' => 
        array (
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Dataset',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 441,
            'endLine' => 441,
            'startColumn' => 29,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Cluster the dataset by assigning a label to each sample.
 *
 * @param Dataset $dataset
 * @throws RuntimeException
 * @return list<int>
 */',
        'startLine' => 441,
        'endLine' => 450,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'predictSample' => 
      array (
        'name' => 'predictSample',
        'parameters' => 
        array (
          'sample' => 
          array (
            'name' => 'sample',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 460,
            'endLine' => 460,
            'startColumn' => 35,
            'endColumn' => 47,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Label a given sample based on its distance from a particular centroid.
 *
 * @internal
 *
 * @param list<int|float> $sample
 * @return int
 */',
        'startLine' => 460,
        'endLine' => 475,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'proba' => 
      array (
        'name' => 'proba',
        'parameters' => 
        array (
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Dataset',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 484,
            'endLine' => 484,
            'startColumn' => 27,
            'endColumn' => 42,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Estimate the joint probabilities for each possible outcome.
 *
 * @param Dataset $dataset
 * @throws RuntimeException
 * @return list<float[]>
 */',
        'startLine' => 484,
        'endLine' => 493,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'probaSample' => 
      array (
        'name' => 'probaSample',
        'parameters' => 
        array (
          'sample' => 
          array (
            'name' => 'sample',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 503,
            'endLine' => 503,
            'startColumn' => 33,
            'endColumn' => 45,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the membership of a sample to each of the k centroids.
 *
 * @internal
 *
 * @param list<int|float> $sample
 * @return float[]
 */',
        'startLine' => 503,
        'endLine' => 522,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      'inertia' => 
      array (
        'name' => 'inertia',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 532,
            'endLine' => 532,
            'startColumn' => 32,
            'endColumn' => 45,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'labels' => 
          array (
            'name' => 'labels',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 532,
            'endLine' => 532,
            'startColumn' => 48,
            'endColumn' => 60,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the average sum of distances between all samples and their closest
 * centroid.
 *
 * @param list<list<int|float>> $samples
 * @param list<int> $labels
 * @return float
 */',
        'startLine' => 532,
        'endLine' => 543,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      '__serialize' => 
      array (
        'name' => '__serialize',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return an associative array containing the data used to serialize the object.
 *
 * @return mixed[]
 */',
        'startLine' => 550,
        'endLine' => 557,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
      '__toString' => 
      array (
        'name' => '__toString',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the string representation of the object.
 *
 * @internal
 *
 * @return string
 */',
        'startLine' => 566,
        'endLine' => 569,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\KMeans',
        'aliasName' => NULL,
      ),
    ),
    'traitsData' => 
    array (
      'aliases' => 
      array (
      ),
      'modifiers' => 
      array (
      ),
      'precedences' => 
      array (
      ),
      'hashes' => 
      array (
      ),
    ),
  ),
));