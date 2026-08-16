<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Clusterers/DBSCAN.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Clusterers\DBSCAN
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-c20f49fa944c116766858742cde7273dad467990e8a254edf0d87aea8a5eae44',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Clusterers/DBSCAN.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Clusterers',
    'name' => 'Rubix\\ML\\Clusterers\\DBSCAN',
    'shortName' => 'DBSCAN',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * DBSCAN
 *
 * *Density-Based Spatial Clustering of Applications with Noise* is a clustering algorithm
 * able to find non-linearly separable and arbitrarily-shaped clusters given a radius and
 * density constraint. In addition, DBSCAN also has the ability to mark outliers as *noise*
 * and thus can be used as a *quasi* anomaly detector.
 *
 * > **Note**: Noise samples are assigned to the cluster number *-1*.
 *
 * References:
 * [1] M. Ester et al. (1996). A Density-Based Algorithm for Discovering Clusters.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 40,
    'endLine' => 214,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'START_CLUSTER' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'name' => 'START_CLUSTER',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 47,
            'endLine' => 47,
            'startTokenPos' => 117,
            'startFilePos' => 1357,
            'endTokenPos' => 117,
            'endFilePos' => 1357,
          ),
        ),
        'docComment' => '/**
 * The starting cluster number.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 47,
        'endLine' => 47,
        'startColumn' => 5,
        'endColumn' => 35,
      ),
      'NOISE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'name' => 'NOISE',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '-1',
          'attributes' => 
          array (
            'startLine' => 54,
            'endLine' => 54,
            'startTokenPos' => 130,
            'startFilePos' => 1478,
            'endTokenPos' => 131,
            'endFilePos' => 1479,
          ),
        ),
        'docComment' => '/**
 * The cluster number assigned to noise samples.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 54,
        'endLine' => 54,
        'startColumn' => 5,
        'endColumn' => 28,
      ),
    ),
    'immediateProperties' => 
    array (
      'radius' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'name' => 'radius',
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
 * The maximum distance between two points to be considered neighbors. The smaller the value,
 * the tighter the clusters will be.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 62,
        'endLine' => 62,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'minDensity' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'name' => 'minDensity',
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
 * The minimum number of points to from a dense region or cluster.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 69,
        'endLine' => 69,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'tree' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'name' => 'tree',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Graph\\Trees\\Spatial',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The spatial tree used to run range searches.
 *
 * @var Spatial
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 76,
        'endLine' => 76,
        'startColumn' => 5,
        'endColumn' => 28,
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
          'radius' => 
          array (
            'name' => 'radius',
            'default' => 
            array (
              'code' => '0.5',
              'attributes' => 
              array (
                'startLine' => 84,
                'endLine' => 84,
                'startTokenPos' => 175,
                'startFilePos' => 2155,
                'endTokenPos' => 175,
                'endFilePos' => 2157,
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
            'startLine' => 84,
            'endLine' => 84,
            'startColumn' => 33,
            'endColumn' => 51,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'minDensity' => 
          array (
            'name' => 'minDensity',
            'default' => 
            array (
              'code' => '5',
              'attributes' => 
              array (
                'startLine' => 84,
                'endLine' => 84,
                'startTokenPos' => 184,
                'startFilePos' => 2178,
                'endTokenPos' => 184,
                'endFilePos' => 2178,
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
            'startLine' => 84,
            'endLine' => 84,
            'startColumn' => 54,
            'endColumn' => 72,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'tree' => 
          array (
            'name' => 'tree',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 84,
                'endLine' => 84,
                'startTokenPos' => 194,
                'startFilePos' => 2198,
                'endTokenPos' => 194,
                'endFilePos' => 2201,
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
                      'name' => 'Rubix\\ML\\Graph\\Trees\\Spatial',
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
            'startLine' => 84,
            'endLine' => 84,
            'startColumn' => 75,
            'endColumn' => 95,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param float $radius
 * @param int $minDensity
 * @param Spatial|null $tree
 * @throws InvalidArgumentException
 */',
        'startLine' => 84,
        'endLine' => 99,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
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
        'startLine' => 106,
        'endLine' => 109,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
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
 * @return list<\\Rubix\\ML\\DataType>
 */',
        'startLine' => 116,
        'endLine' => 119,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
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
        'startLine' => 126,
        'endLine' => 133,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
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
            'startLine' => 141,
            'endLine' => 141,
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
 * Make predictions from a dataset.
 *
 * @param Dataset $dataset
 * @return list<int>
 */',
        'startLine' => 141,
        'endLine' => 201,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
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
        'startLine' => 210,
        'endLine' => 213,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\DBSCAN',
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