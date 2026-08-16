<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Regressors/RadiusNeighborsRegressor.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Regressors\RadiusNeighborsRegressor
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-a9c0381b35ed2f3d9e8baaad1d0e50cc8cea542305eb97691c94307ae972fd20',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Regressors/RadiusNeighborsRegressor.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Regressors',
    'name' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
    'shortName' => 'RadiusNeighborsRegressor',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Radius Neighbors Regressor
 *
 * This is the regressor version of Radius Neighbors implementing a binary spatial tree under
 * the hood for fast radius queries. The prediction is a weighted average of each label from
 * the training set that is within a fixed user-defined radius.
 *
 * > **Note**: Unknown samples with no training samples within radius are labeled
 * *NaN*. As such, Radius Neighbors is also a quasi anomaly detector.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 41,
    'endLine' => 232,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\Persistable',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AutotrackRevisions',
    ),
    'immediateConstants' => 
    array (
      'OUTLIER_VALUE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'name' => 'OUTLIER_VALUE',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => 'NAN',
          'attributes' => 
          array (
            'startLine' => 50,
            'endLine' => 50,
            'startTokenPos' => 140,
            'startFilePos' => 1627,
            'endTokenPos' => 140,
            'endFilePos' => 1629,
          ),
        ),
        'docComment' => '/**
 * The value to assign to outliers when making a prediction.
 *
 * @var mixed
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 50,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 37,
      ),
    ),
    'immediateProperties' => 
    array (
      'radius' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
 * The radius within which points are considered neighbors.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 57,
        'endLine' => 57,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'weighted' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'name' => 'weighted',
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
 * Should we consider the distances of our nearest neighbors when making predictions?
 *
 * @var bool
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 64,
        'endLine' => 64,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'tree' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
        'startLine' => 71,
        'endLine' => 71,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'featureCount' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
            'startLine' => 78,
            'endLine' => 78,
            'startTokenPos' => 181,
            'startFilePos' => 2180,
            'endTokenPos' => 181,
            'endFilePos' => 2183,
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
        'startLine' => 78,
        'endLine' => 78,
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
          'radius' => 
          array (
            'name' => 'radius',
            'default' => 
            array (
              'code' => '1.0',
              'attributes' => 
              array (
                'startLine' => 86,
                'endLine' => 86,
                'startTokenPos' => 198,
                'startFilePos' => 2381,
                'endTokenPos' => 198,
                'endFilePos' => 2383,
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
            'startLine' => 86,
            'endLine' => 86,
            'startColumn' => 33,
            'endColumn' => 51,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'weighted' => 
          array (
            'name' => 'weighted',
            'default' => 
            array (
              'code' => 'false',
              'attributes' => 
              array (
                'startLine' => 86,
                'endLine' => 86,
                'startTokenPos' => 207,
                'startFilePos' => 2403,
                'endTokenPos' => 207,
                'endFilePos' => 2407,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'bool',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 86,
            'endLine' => 86,
            'startColumn' => 54,
            'endColumn' => 75,
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
                'startLine' => 86,
                'endLine' => 86,
                'startTokenPos' => 217,
                'startFilePos' => 2427,
                'endTokenPos' => 217,
                'endFilePos' => 2430,
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
            'startLine' => 86,
            'endLine' => 86,
            'startColumn' => 78,
            'endColumn' => 98,
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
 * @param bool $weighted
 * @param Spatial|null $tree
 * @throws InvalidArgumentException
 */',
        'startLine' => 86,
        'endLine' => 96,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
        'startLine' => 105,
        'endLine' => 108,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
        'startLine' => 117,
        'endLine' => 120,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
        'startLine' => 129,
        'endLine' => 136,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
        'startLine' => 143,
        'endLine' => 146,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
        'startLine' => 153,
        'endLine' => 156,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
            'startLine' => 163,
            'endLine' => 163,
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
 * @param Labeled $dataset
 */',
        'startLine' => 163,
        'endLine' => 175,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
            'startLine' => 184,
            'endLine' => 184,
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
 * Make a prediction based on the nearest neighbors.
 *
 * @param Dataset $dataset
 * @throws RuntimeException
 * @return list<int|float>
 */',
        'startLine' => 184,
        'endLine' => 193,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
            'startLine' => 203,
            'endLine' => 203,
            'startColumn' => 35,
            'endColumn' => 47,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
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
                  'name' => 'int',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'float',
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
 * Predict a single sample and return the result.
 *
 * @internal
 *
 * @param list<string|int|float> $sample
 * @return int|float
 */',
        'startLine' => 203,
        'endLine' => 219,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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
        'startLine' => 228,
        'endLine' => 231,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\RadiusNeighborsRegressor',
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