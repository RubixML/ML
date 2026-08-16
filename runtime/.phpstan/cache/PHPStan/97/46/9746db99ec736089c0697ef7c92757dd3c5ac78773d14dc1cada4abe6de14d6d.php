<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/AnomalyDetectors/LocalOutlierFactor.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\AnomalyDetectors\LocalOutlierFactor
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-12a0b4783f2c018f59cb0cd356ee37ed7ac962bd374db60d98228c7fa9ba694f',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/AnomalyDetectors/LocalOutlierFactor.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\AnomalyDetectors',
    'name' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
    'shortName' => 'LocalOutlierFactor',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Local Outlier Factor
 *
 * Local Outlier Factor (LOF) measures the local deviation of density of an unknown
 * sample with respect to its *k* nearest neighbors from the training set. As such,
 * LOF only considers the local region (or *neighborhood*) of an unknown sample
 * which enables it to detect anomalies within individual clusters of data.
 *
 * References:
 * [1] M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 40,
    'endLine' => 333,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\AnomalyDetectors\\Scoring',
      3 => 'Rubix\\ML\\Persistable',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AutotrackRevisions',
    ),
    'immediateConstants' => 
    array (
      'DEFAULT_THRESHOLD' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'name' => 'DEFAULT_THRESHOLD',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1.5',
          'attributes' => 
          array (
            'startLine' => 49,
            'endLine' => 49,
            'startTokenPos' => 135,
            'startFilePos' => 1515,
            'endTokenPos' => 135,
            'endFilePos' => 1517,
          ),
        ),
        'docComment' => '/**
 * The default minimum anomaly score for a sample to be flagged.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 49,
        'endLine' => 49,
        'startColumn' => 5,
        'endColumn' => 44,
      ),
    ),
    'immediateProperties' => 
    array (
      'k' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
 * The number of nearest neighbors to consider a local region.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 56,
        'endLine' => 56,
        'startColumn' => 5,
        'endColumn' => 21,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'contamination' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'name' => 'contamination',
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
                  'name' => 'float',
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
            'startLine' => 63,
            'endLine' => 63,
            'startTokenPos' => 158,
            'startFilePos' => 1820,
            'endTokenPos' => 158,
            'endFilePos' => 1823,
          ),
        ),
        'docComment' => '/**
 * The percentage of outliers that are assumed to be present in the training set.
 *
 * @var float|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 63,
        'endLine' => 63,
        'startColumn' => 5,
        'endColumn' => 43,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'tree' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
 * The k-d tree used for nearest neighbor queries.
 *
 * @var Spatial
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 70,
        'endLine' => 70,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'kdistances' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'name' => 'kdistances',
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
            'startLine' => 77,
            'endLine' => 79,
            'startTokenPos' => 180,
            'startFilePos' => 2127,
            'endTokenPos' => 184,
            'endFilePos' => 2144,
          ),
        ),
        'docComment' => '/**
 * The precomputed k distances between each training sample and its k\'th nearest neighbor.
 *
 * @var float[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 77,
        'endLine' => 79,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'lrds' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'name' => 'lrds',
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
            'startLine' => 86,
            'endLine' => 88,
            'startTokenPos' => 197,
            'startFilePos' => 2292,
            'endTokenPos' => 201,
            'endFilePos' => 2309,
          ),
        ),
        'docComment' => '/**
 * The precomputed local reachability densities of the training set.
 *
 * @var float[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 86,
        'endLine' => 88,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'threshold' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'name' => 'threshold',
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
                  'name' => 'float',
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
            'startLine' => 95,
            'endLine' => 95,
            'startTokenPos' => 215,
            'startFilePos' => 2466,
            'endTokenPos' => 215,
            'endFilePos' => 2469,
          ),
        ),
        'docComment' => '/**
 * The local outlier factor threshold used by the decision function.
 *
 * @var float|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 95,
        'endLine' => 95,
        'startColumn' => 5,
        'endColumn' => 39,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'featureCount' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'name' => 'featureCount',
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
                  'name' => 'int',
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
            'startLine' => 102,
            'endLine' => 102,
            'startTokenPos' => 229,
            'startFilePos' => 2599,
            'endTokenPos' => 229,
            'endFilePos' => 2602,
          ),
        ),
        'docComment' => '/**
 * The dimensionality of the training set.
 *
 * @var int|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 102,
        'endLine' => 102,
        'startColumn' => 5,
        'endColumn' => 40,
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
            'default' => 
            array (
              'code' => '20',
              'attributes' => 
              array (
                'startLine' => 110,
                'endLine' => 110,
                'startTokenPos' => 246,
                'startFilePos' => 2797,
                'endTokenPos' => 246,
                'endFilePos' => 2798,
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
            'startLine' => 110,
            'endLine' => 110,
            'startColumn' => 33,
            'endColumn' => 43,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'contamination' => 
          array (
            'name' => 'contamination',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 110,
                'endLine' => 110,
                'startTokenPos' => 256,
                'startFilePos' => 2825,
                'endTokenPos' => 256,
                'endFilePos' => 2828,
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
                      'name' => 'float',
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
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 110,
            'endLine' => 110,
            'startColumn' => 46,
            'endColumn' => 73,
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
                'startLine' => 110,
                'endLine' => 110,
                'startTokenPos' => 266,
                'startFilePos' => 2848,
                'endTokenPos' => 266,
                'endFilePos' => 2851,
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
            'startLine' => 110,
            'endLine' => 110,
            'startColumn' => 76,
            'endColumn' => 96,
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
 * @param int $k
 * @param float|null $contamination
 * @param Spatial|null $tree
 * @throws InvalidArgumentException
 */',
        'startLine' => 110,
        'endLine' => 125,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
 * @internal
 *
 * @return EstimatorType
 */',
        'startLine' => 134,
        'endLine' => 137,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
 * @internal
 *
 * @return list<\\Rubix\\ML\\DataType>
 */',
        'startLine' => 146,
        'endLine' => 149,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
 * @internal
 *
 * @return mixed[]
 */',
        'startLine' => 158,
        'endLine' => 165,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
        'startLine' => 172,
        'endLine' => 175,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'aliasName' => NULL,
      ),
      'tree' => 
      array (
        'name' => 'tree',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Graph\\Trees\\Spatial',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the base spatial tree instance.
 *
 * @return Spatial
 */',
        'startLine' => 182,
        'endLine' => 185,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
            'startLine' => 192,
            'endLine' => 192,
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
        'startLine' => 192,
        'endLine' => 229,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
            'startLine' => 237,
            'endLine' => 237,
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
        'startLine' => 237,
        'endLine' => 246,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
            'startLine' => 256,
            'endLine' => 256,
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
 * Predict a single sample and return the result.
 *
 * @internal
 *
 * @param list<int|float> $sample
 * @return int
 */',
        'startLine' => 256,
        'endLine' => 259,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'aliasName' => NULL,
      ),
      'score' => 
      array (
        'name' => 'score',
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
            'startLine' => 268,
            'endLine' => 268,
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
 * Return the anomaly scores assigned to the samples in a dataset.
 *
 * @param Dataset $dataset
 * @throws RuntimeException
 * @return list<float>
 */',
        'startLine' => 268,
        'endLine' => 277,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'aliasName' => NULL,
      ),
      'localOutlierFactor' => 
      array (
        'name' => 'localOutlierFactor',
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
            'startLine' => 286,
            'endLine' => 286,
            'startColumn' => 43,
            'endColumn' => 55,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the local outlier factor of a given sample given its k
 * nearest neighbors.
 *
 * @param list<int|float> $sample
 * @return float
 */',
        'startLine' => 286,
        'endLine' => 299,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'aliasName' => NULL,
      ),
      'localReachabilityDensity' => 
      array (
        'name' => 'localReachabilityDensity',
        'parameters' => 
        array (
          'indices' => 
          array (
            'name' => 'indices',
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
            'startLine' => 309,
            'endLine' => 309,
            'startColumn' => 49,
            'endColumn' => 62,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'distances' => 
          array (
            'name' => 'distances',
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
            'startLine' => 309,
            'endLine' => 309,
            'startColumn' => 65,
            'endColumn' => 80,
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
 * Calculate the local reachability density of a sample given its
 * distances to its k nearest neighbors.
 *
 * @param list<string|int|float> $indices
 * @param list<float> $distances
 * @return float
 */',
        'startLine' => 309,
        'endLine' => 320,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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
        'startLine' => 329,
        'endLine' => 332,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\LocalOutlierFactor',
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