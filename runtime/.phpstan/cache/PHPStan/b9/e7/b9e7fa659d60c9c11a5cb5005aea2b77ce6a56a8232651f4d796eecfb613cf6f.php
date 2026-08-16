<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/AnomalyDetectors/GaussianMLE.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\AnomalyDetectors\GaussianMLE
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-ccdd995f9aaacc634fda2bd63bbd00a0495ba3b5f13cba6e6268a2cb83420be8',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/AnomalyDetectors/GaussianMLE.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\AnomalyDetectors',
    'name' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
    'shortName' => 'GaussianMLE',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Gaussian MLE
 *
 * The Gaussian Maximum Likelihood Estimator (MLE) is able to spot outliers by computing
 * a probability density function (PDF) over the features assuming they are independently
 * and normally (Gaussian) distributed. Samples that are assigned low probability density
 * are more likely to be outliers.
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing
 * Sample Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 41,
    'endLine' => 370,
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
    ),
    'immediateProperties' => 
    array (
      'contamination' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
        'startLine' => 50,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 35,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'smoothing' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'name' => 'smoothing',
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
 * The amount of epsilon smoothing added to the variance of each feature.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 57,
        'endLine' => 57,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'means' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'name' => 'means',
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
            'startLine' => 64,
            'endLine' => 66,
            'startTokenPos' => 156,
            'startFilePos' => 1798,
            'endTokenPos' => 160,
            'endFilePos' => 1815,
          ),
        ),
        'docComment' => '/**
 * The precomputed means of each feature column of the training set.
 *
 * @var float[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 64,
        'endLine' => 66,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'variances' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'name' => 'variances',
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
            'startLine' => 73,
            'endLine' => 75,
            'startTokenPos' => 173,
            'startFilePos' => 1972,
            'endTokenPos' => 177,
            'endFilePos' => 1989,
          ),
        ),
        'docComment' => '/**
 * The precomputed variances of each feature column of the training set.
 *
 * @var float[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 73,
        'endLine' => 75,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'epsilon' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'name' => 'epsilon',
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
            'startLine' => 82,
            'endLine' => 82,
            'startTokenPos' => 191,
            'startFilePos' => 2128,
            'endTokenPos' => 191,
            'endFilePos' => 2131,
          ),
        ),
        'docComment' => '/**
 * A small portion of variance to add for smoothing.
 *
 * @var float|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 82,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 37,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'n' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
            'startLine' => 89,
            'endLine' => 89,
            'startTokenPos' => 204,
            'startFilePos' => 2268,
            'endTokenPos' => 204,
            'endFilePos' => 2268,
          ),
        ),
        'docComment' => '/**
 * The number of samples that have passed through training so far.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 89,
        'endLine' => 89,
        'startColumn' => 5,
        'endColumn' => 25,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'threshold' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
            'startLine' => 96,
            'endLine' => 96,
            'startTokenPos' => 218,
            'startFilePos' => 2422,
            'endTokenPos' => 218,
            'endFilePos' => 2425,
          ),
        ),
        'docComment' => '/**
 * The minimum log likelihood score necessary to flag an anomaly.
 *
 * @var float|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 96,
        'endLine' => 96,
        'startColumn' => 5,
        'endColumn' => 39,
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
                'startLine' => 103,
                'endLine' => 103,
                'startTokenPos' => 235,
                'startFilePos' => 2606,
                'endTokenPos' => 235,
                'endFilePos' => 2608,
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
            'startLine' => 103,
            'endLine' => 103,
            'startColumn' => 33,
            'endColumn' => 58,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'smoothing' => 
          array (
            'name' => 'smoothing',
            'default' => 
            array (
              'code' => '1.0E-9',
              'attributes' => 
              array (
                'startLine' => 103,
                'endLine' => 103,
                'startTokenPos' => 244,
                'startFilePos' => 2630,
                'endTokenPos' => 244,
                'endFilePos' => 2633,
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
            'startLine' => 103,
            'endLine' => 103,
            'startColumn' => 61,
            'endColumn' => 83,
            'parameterIndex' => 1,
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
 * @param float $smoothing
 * @throws InvalidArgumentException
 */',
        'startLine' => 103,
        'endLine' => 117,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
        'startLine' => 126,
        'endLine' => 129,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
        'startLine' => 138,
        'endLine' => 143,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
        'startLine' => 152,
        'endLine' => 158,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
        'startLine' => 165,
        'endLine' => 168,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'aliasName' => NULL,
      ),
      'means' => 
      array (
        'name' => 'means',
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
 * Return the column means computed from the training set.
 *
 * @return float[]
 */',
        'startLine' => 175,
        'endLine' => 178,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'aliasName' => NULL,
      ),
      'variances' => 
      array (
        'name' => 'variances',
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
 * Return the column variances computed from the training set.
 *
 * @return float[]
 */',
        'startLine' => 185,
        'endLine' => 188,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
            'startLine' => 195,
            'endLine' => 195,
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
        'startLine' => 195,
        'endLine' => 226,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
            'startLine' => 233,
            'endLine' => 233,
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
        'startLine' => 233,
        'endLine' => 286,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
            'startLine' => 294,
            'endLine' => 294,
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
        'startLine' => 294,
        'endLine' => 303,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
            'startLine' => 313,
            'endLine' => 313,
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
        'startLine' => 313,
        'endLine' => 316,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
            'startLine' => 325,
            'endLine' => 325,
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
        'startLine' => 325,
        'endLine' => 334,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'aliasName' => NULL,
      ),
      'logLikelihood' => 
      array (
        'name' => 'logLikelihood',
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
            'startLine' => 342,
            'endLine' => 342,
            'startColumn' => 38,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the log likelihood of a sample being an outlier.
 *
 * @param list<int|float> $sample
 * @return float
 */',
        'startLine' => 342,
        'endLine' => 357,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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
        'startLine' => 366,
        'endLine' => 369,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\GaussianMLE',
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