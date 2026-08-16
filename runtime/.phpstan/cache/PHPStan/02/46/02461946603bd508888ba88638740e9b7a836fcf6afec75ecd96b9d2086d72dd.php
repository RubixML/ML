<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/AnomalyDetectors/Loda.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\AnomalyDetectors\Loda
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-ccc1586a47209066e918f353ac590a5baef9ee434945930eaaa00967576b62c6',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/AnomalyDetectors/Loda.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\AnomalyDetectors',
    'name' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
    'shortName' => 'Loda',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Loda
 *
 * *Lightweight Online Detector of Anomalies* uses a series of sparse random
 * projection vectors to produce scalar inputs to an ensemble of unique
 * one-dimensional equi-width histograms. The histograms are then used to estimate
 * the probability density of an unknown sample during inference.
 *
 * References:
 * [1] T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
 * [2] L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 52,
    'endLine' => 421,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\Online',
      3 => 'Rubix\\ML\\AnomalyDetectors\\Scoring',
      4 => 'Rubix\\ML\\Persistable',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AutotrackRevisions',
    ),
    'immediateConstants' => 
    array (
      'MIN_BINS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'MIN_BINS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '3',
          'attributes' => 
          array (
            'startLine' => 61,
            'endLine' => 61,
            'startTokenPos' => 208,
            'startFilePos' => 1697,
            'endTokenPos' => 208,
            'endFilePos' => 1697,
          ),
        ),
        'docComment' => '/**
 * The minimum number of histogram bins.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 61,
        'endLine' => 61,
        'startColumn' => 5,
        'endColumn' => 33,
      ),
      'MIN_SPARSE_DIMENSIONS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'MIN_SPARSE_DIMENSIONS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '3',
          'attributes' => 
          array (
            'startLine' => 68,
            'endLine' => 68,
            'startTokenPos' => 221,
            'startFilePos' => 1858,
            'endTokenPos' => 221,
            'endFilePos' => 1858,
          ),
        ),
        'docComment' => '/**
 * The minimum dimensionality required to produce sparse projections.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 68,
        'endLine' => 68,
        'startColumn' => 5,
        'endColumn' => 46,
      ),
    ),
    'immediateProperties' => 
    array (
      'contamination' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'contamination',
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
 * The proportion of outliers that are assumed to be present in the training set.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 75,
        'endLine' => 75,
        'startColumn' => 5,
        'endColumn' => 35,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'estimators' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'estimators',
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
 * The number of projection/histogram pairs in the ensemble.
 *
 * @var positive-int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 82,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'bins' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'bins',
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
            'startLine' => 89,
            'endLine' => 89,
            'startTokenPos' => 253,
            'startFilePos' => 2298,
            'endTokenPos' => 253,
            'endFilePos' => 2301,
          ),
        ),
        'docComment' => '/**
 * The number of bins in each equi-width histogram.
 *
 * @var int|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 89,
        'endLine' => 89,
        'startColumn' => 5,
        'endColumn' => 32,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'fitBins' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'fitBins',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * Should we calculate the equi-width bin count on the fly?
 *
 * @var bool
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 96,
        'endLine' => 96,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'r' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'r',
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
                  'name' => 'Tensor\\Matrix',
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
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 103,
            'endLine' => 103,
            'startTokenPos' => 276,
            'startFilePos' => 2557,
            'endTokenPos' => 276,
            'endFilePos' => 2560,
          ),
        ),
        'docComment' => '/**
 * The sparse random projection matrix.
 *
 * @var Matrix|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 103,
        'endLine' => 103,
        'startColumn' => 5,
        'endColumn' => 32,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'histograms' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'histograms',
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
            'startLine' => 110,
            'endLine' => 112,
            'startTokenPos' => 289,
            'startFilePos' => 2728,
            'endTokenPos' => 293,
            'endFilePos' => 2745,
          ),
        ),
        'docComment' => '/**
 * The edges and bin counts of each histogram.
 *
 * @var array{list<float>,list<int<0,max>>}|mixed[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 110,
        'endLine' => 112,
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
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
        'default' => NULL,
        'docComment' => '/**
 * The minimum negative log likelihood score necessary to flag an anomaly.
 *
 * @var float|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 119,
        'endLine' => 119,
        'startColumn' => 5,
        'endColumn' => 32,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'n' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'name' => 'n',
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
        'default' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 126,
            'endLine' => 126,
            'startTokenPos' => 316,
            'startFilePos' => 3030,
            'endTokenPos' => 316,
            'endFilePos' => 3030,
          ),
        ),
        'docComment' => '/**
 * The number of samples that have been learned so far.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 126,
        'endLine' => 126,
        'startColumn' => 5,
        'endColumn' => 25,
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
          'contamination' => 
          array (
            'name' => 'contamination',
            'default' => 
            array (
              'code' => '0.1',
              'attributes' => 
              array (
                'startLine' => 134,
                'endLine' => 134,
                'startTokenPos' => 333,
                'startFilePos' => 3239,
                'endTokenPos' => 333,
                'endFilePos' => 3241,
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
            'startLine' => 134,
            'endLine' => 134,
            'startColumn' => 33,
            'endColumn' => 58,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'estimators' => 
          array (
            'name' => 'estimators',
            'default' => 
            array (
              'code' => '100',
              'attributes' => 
              array (
                'startLine' => 134,
                'endLine' => 134,
                'startTokenPos' => 342,
                'startFilePos' => 3262,
                'endTokenPos' => 342,
                'endFilePos' => 3264,
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
            'startLine' => 134,
            'endLine' => 134,
            'startColumn' => 61,
            'endColumn' => 81,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'bins' => 
          array (
            'name' => 'bins',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 134,
                'endLine' => 134,
                'startTokenPos' => 352,
                'startFilePos' => 3280,
                'endTokenPos' => 352,
                'endFilePos' => 3283,
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
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 134,
            'endLine' => 134,
            'startColumn' => 84,
            'endColumn' => 100,
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
 * @param float $contamination
 * @param int $estimators
 * @param int|null $bins
 * @throws InvalidArgumentException
 */',
        'startLine' => 134,
        'endLine' => 155,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
 * Return the integer encoded estimator type.
 *
 * @internal
 *
 * @return EstimatorType
 */',
        'startLine' => 164,
        'endLine' => 167,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
 * @return list<DataType>
 */',
        'startLine' => 176,
        'endLine' => 181,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
        'startLine' => 190,
        'endLine' => 197,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
        'startLine' => 204,
        'endLine' => 207,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
            'startLine' => 214,
            'endLine' => 214,
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
        'startLine' => 214,
        'endLine' => 272,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
            'startLine' => 279,
            'endLine' => 279,
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
        'startLine' => 279,
        'endLine' => 325,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
            'startLine' => 333,
            'endLine' => 333,
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
 * @return array<float>
 */',
        'startLine' => 333,
        'endLine' => 336,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
            'startLine' => 345,
            'endLine' => 345,
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
 * @return array<float>
 */',
        'startLine' => 345,
        'endLine' => 359,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'aliasName' => NULL,
      ),
      'densities' => 
      array (
        'name' => 'densities',
        'parameters' => 
        array (
          'projections' => 
          array (
            'name' => 'projections',
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
            'startLine' => 368,
            'endLine' => 368,
            'startColumn' => 34,
            'endColumn' => 51,
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
 * Estimate the probability density function of each 1-dimensional projection using the histograms
 * created during training.
 *
 * @param list<list<float>> $projections
 * @return array<float>
 */',
        'startLine' => 368,
        'endLine' => 397,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'aliasName' => NULL,
      ),
      'decide' => 
      array (
        'name' => 'decide',
        'parameters' => 
        array (
          'score' => 
          array (
            'name' => 'score',
            'default' => NULL,
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
            'startLine' => 405,
            'endLine' => 405,
            'startColumn' => 31,
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
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * The decision function.
 *
 * @param float $score
 * @return int
 */',
        'startLine' => 405,
        'endLine' => 408,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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
        'startLine' => 417,
        'endLine' => 420,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Loda',
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