<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/BootstrapAggregator.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\BootstrapAggregator
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-f2bdbf321b70610ace9302d5efa1fd9d17a538f851b7327dbdfd9e2a9ab1087d',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\BootstrapAggregator',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/BootstrapAggregator.php',
      ),
    ),
    'namespace' => 'Rubix\\ML',
    'name' => 'Rubix\\ML\\BootstrapAggregator',
    'shortName' => 'BootstrapAggregator',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Bootstrap Aggregator
 *
 * Bootstrap Aggregating (or *bagging* for short) is a model averaging technique designed
 * to improve the stability and performance of a user-specified base estimator by training
 * a number of them on a unique *bootstrapped* training set sampled at random with
 * replacement.
 *
 * References:
 * [1] L. Breiman. (1996). Bagging Predictors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 40,
    'endLine' => 265,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\Parallel',
      3 => 'Rubix\\ML\\Persistable',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AutotrackRevisions',
      1 => 'Rubix\\ML\\Traits\\Multiprocessing',
    ),
    'immediateConstants' => 
    array (
      'COMPATIBLE_ESTIMATOR_TYPES' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'name' => 'COMPATIBLE_ESTIMATOR_TYPES',
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
        'value' => 
        array (
          'code' => '[\\Rubix\\ML\\EstimatorType::CLASSIFIER, \\Rubix\\ML\\EstimatorType::REGRESSOR, \\Rubix\\ML\\EstimatorType::ANOMALY_DETECTOR]',
          'attributes' => 
          array (
            'startLine' => 49,
            'endLine' => 53,
            'startTokenPos' => 142,
            'startFilePos' => 1516,
            'endTokenPos' => 159,
            'endFilePos' => 1632,
          ),
        ),
        'docComment' => '/**
 * The estimator type codes that the ensemble is compatible with.
 *
 * @var list<int>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 49,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 6,
      ),
      'MIN_SUBSAMPLE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'name' => 'MIN_SUBSAMPLE',
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
        'value' => 
        array (
          'code' => '1',
          'attributes' => 
          array (
            'startLine' => 58,
            'endLine' => 58,
            'startTokenPos' => 174,
            'startFilePos' => 1741,
            'endTokenPos' => 174,
            'endFilePos' => 1741,
          ),
        ),
        'docComment' => '/**
 * The minimum size of each training subset.
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 58,
        'endLine' => 58,
        'startColumn' => 5,
        'endColumn' => 42,
      ),
    ),
    'immediateProperties' => 
    array (
      'base' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'name' => 'base',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Learner',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The base learner.
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 63,
        'endLine' => 63,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'estimators' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
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
 * The number of base learners to train in the ensemble.
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 68,
        'endLine' => 68,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'ratio' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'name' => 'ratio',
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
 * The ratio of samples from the training set to randomly subsample to train each base learner.
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 73,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'ensemble' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'name' => 'ensemble',
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
            'startLine' => 80,
            'endLine' => 82,
            'startTokenPos' => 214,
            'startFilePos' => 2186,
            'endTokenPos' => 218,
            'endFilePos' => 2203,
          ),
        ),
        'docComment' => '/**
 * The ensemble of estimators.
 *
 * @var list<Learner>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 80,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 6,
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
          'base' => 
          array (
            'name' => 'base',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Learner',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 90,
            'endLine' => 90,
            'startColumn' => 33,
            'endColumn' => 45,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'estimators' => 
          array (
            'name' => 'estimators',
            'default' => 
            array (
              'code' => '10',
              'attributes' => 
              array (
                'startLine' => 90,
                'endLine' => 90,
                'startTokenPos' => 240,
                'startFilePos' => 2413,
                'endTokenPos' => 240,
                'endFilePos' => 2414,
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
            'startLine' => 90,
            'endLine' => 90,
            'startColumn' => 48,
            'endColumn' => 67,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'ratio' => 
          array (
            'name' => 'ratio',
            'default' => 
            array (
              'code' => '0.5',
              'attributes' => 
              array (
                'startLine' => 90,
                'endLine' => 90,
                'startTokenPos' => 249,
                'startFilePos' => 2432,
                'endTokenPos' => 249,
                'endFilePos' => 2434,
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
            'startLine' => 90,
            'endLine' => 90,
            'startColumn' => 70,
            'endColumn' => 87,
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
 * @param Learner $base
 * @param int $estimators
 * @param float $ratio
 * @throws InvalidArgumentException
 */',
        'startLine' => 90,
        'endLine' => 112,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
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
        'startLine' => 121,
        'endLine' => 124,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
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
        'startLine' => 133,
        'endLine' => 136,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
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
 * @return array
 */',
        'startLine' => 145,
        'endLine' => 152,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
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
        'startLine' => 159,
        'endLine' => 162,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
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
            'startLine' => 171,
            'endLine' => 171,
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
 * Instantiate and train each base estimator in the ensemble on a bootstrap
 * training set.
 *
 * @param Dataset $dataset
 * @throws InvalidArgumentException
 */',
        'startLine' => 171,
        'endLine' => 205,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
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
            'startLine' => 214,
            'endLine' => 214,
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
 * @throws RuntimeException
 * @return array
 */',
        'startLine' => 214,
        'endLine' => 238,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'aliasName' => NULL,
      ),
      'decideDiscrete' => 
      array (
        'name' => 'decideDiscrete',
        'parameters' => 
        array (
          'votes' => 
          array (
            'name' => 'votes',
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
            'startLine' => 246,
            'endLine' => 246,
            'startColumn' => 39,
            'endColumn' => 50,
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
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Decide on a discrete-valued outcome.
 *
 * @param string[] $votes
 * @return string
 */',
        'startLine' => 246,
        'endLine' => 252,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
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
        'startLine' => 261,
        'endLine' => 264,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'implementingClassName' => 'Rubix\\ML\\BootstrapAggregator',
        'currentClassName' => 'Rubix\\ML\\BootstrapAggregator',
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