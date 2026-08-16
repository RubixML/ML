<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Classifiers/GaussianNB.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Classifiers\GaussianNB
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-86bce8bee5f35a1d1e32a3d1297deecd0cad8ca59d1d05be7593ab7527e0610f',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Classifiers/GaussianNB.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Classifiers',
    'name' => 'Rubix\\ML\\Classifiers\\GaussianNB',
    'shortName' => 'GaussianNB',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Gaussian Naive Bayes
 *
 * Gaussian Naive Bayes is a version of the Naive Bayes classifier for continuous features. It places
 * a probability density function over the input features on a class basis and uses Bayes\' Theorem to
 * derive the class probabilities. In addition to feature independence, Gaussian NB comes with the
 * additional assumption that all features are normally (Gaussian) distributed.
 *
 * References:
 * [1] T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample
 * Variances.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 51,
    'endLine' => 439,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\Online',
      3 => 'Rubix\\ML\\Probabilistic',
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
      'logPriors' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'name' => 'logPriors',
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
            'startLine' => 60,
            'endLine' => 60,
            'startTokenPos' => 196,
            'startFilePos' => 1836,
            'endTokenPos' => 196,
            'endFilePos' => 1839,
          ),
        ),
        'docComment' => '/**
 * The class prior log probabilities.
 *
 * @var float[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 60,
        'endLine' => 60,
        'startColumn' => 5,
        'endColumn' => 39,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'fitPriors' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'name' => 'fitPriors',
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
 * Should we compute the prior probabilities from the training set?
 *
 * @var bool
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 67,
        'endLine' => 67,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'smoothing' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
        'startLine' => 74,
        'endLine' => 74,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'weights' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'name' => 'weights',
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
            'startLine' => 81,
            'endLine' => 83,
            'startTokenPos' => 227,
            'startFilePos' => 2289,
            'endTokenPos' => 231,
            'endFilePos' => 2306,
          ),
        ),
        'docComment' => '/**
 * The weight of each class as a proportion of the entire training set.
 *
 * @var float[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 81,
        'endLine' => 83,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'means' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 90,
            'endLine' => 92,
            'startTokenPos' => 244,
            'startFilePos' => 2476,
            'endTokenPos' => 248,
            'endFilePos' => 2493,
          ),
        ),
        'docComment' => '/**
 * The means of each feature of the training set conditioned on a class basis.
 *
 * @var array<list<float>>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 90,
        'endLine' => 92,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 99,
            'endLine' => 101,
            'startTokenPos' => 261,
            'startFilePos' => 2663,
            'endTokenPos' => 265,
            'endFilePos' => 2680,
          ),
        ),
        'docComment' => '/**
 * The variances of each feature of the training set conditioned by class.
 *
 * @var array<list<float>>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 99,
        'endLine' => 101,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 108,
            'endLine' => 108,
            'startTokenPos' => 279,
            'startFilePos' => 2819,
            'endTokenPos' => 279,
            'endFilePos' => 2822,
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
        'startLine' => 108,
        'endLine' => 108,
        'startColumn' => 5,
        'endColumn' => 37,
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
          'priors' => 
          array (
            'name' => 'priors',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 115,
                'endLine' => 115,
                'startTokenPos' => 297,
                'startFilePos' => 2997,
                'endTokenPos' => 297,
                'endFilePos' => 3000,
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
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 115,
            'endLine' => 115,
            'startColumn' => 33,
            'endColumn' => 53,
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
                'startLine' => 115,
                'endLine' => 115,
                'startTokenPos' => 306,
                'startFilePos' => 3022,
                'endTokenPos' => 306,
                'endFilePos' => 3025,
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
            'startLine' => 115,
            'endLine' => 115,
            'startColumn' => 56,
            'endColumn' => 78,
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
 * @param float[]|null $priors
 * @param float $smoothing
 * @throws InvalidArgumentException
 */',
        'startLine' => 115,
        'endLine' => 145,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
        'startLine' => 154,
        'endLine' => 157,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
        'startLine' => 166,
        'endLine' => 171,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
        'startLine' => 180,
        'endLine' => 186,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
        'startLine' => 193,
        'endLine' => 196,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'aliasName' => NULL,
      ),
      'priors' => 
      array (
        'name' => 'priors',
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
 * Return the class prior probabilities.
 *
 * @return float[]|null
 */',
        'startLine' => 203,
        'endLine' => 206,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
 * Return the running means of each feature column of the training data by class.
 *
 * @return array<list<float>>|null
 */',
        'startLine' => 213,
        'endLine' => 216,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
 * Return the running variances of each feature column of the training data by class.
 *
 * @return array<list<float>>|null
 */',
        'startLine' => 223,
        'endLine' => 226,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 233,
            'endLine' => 233,
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
 * @param \\Rubix\\ML\\Datasets\\Labeled $dataset
 */',
        'startLine' => 233,
        'endLine' => 238,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 245,
            'endLine' => 245,
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
 * @param \\Rubix\\ML\\Datasets\\Labeled $dataset
 */',
        'startLine' => 245,
        'endLine' => 323,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 332,
            'endLine' => 332,
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
 * Calculate the likelihood of the sample being a member of a class and choose the class with the highest likelihood as the prediction.
 *
 * @param Dataset $dataset
 * @throws RuntimeException
 * @return list<string>
 */',
        'startLine' => 332,
        'endLine' => 341,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 351,
            'endLine' => 351,
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
            'name' => 'string',
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
 * @param (int|float)[] $sample
 * @return string
 */',
        'startLine' => 351,
        'endLine' => 354,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 363,
            'endLine' => 363,
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
 * @return list<array<string,float>>
 */',
        'startLine' => 363,
        'endLine' => 372,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
            'startLine' => 382,
            'endLine' => 382,
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
 * Predict the probabilities of a single sample and return the joint distribution.
 *
 * @internal
 *
 * @param (int|float)[] $sample
 * @return float[]
 */',
        'startLine' => 382,
        'endLine' => 395,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'aliasName' => NULL,
      ),
      'jointLogLikelihood' => 
      array (
        'name' => 'jointLogLikelihood',
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
            'startLine' => 403,
            'endLine' => 403,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the joint log likelihood of a sample being a member of each class.
 *
 * @param list<int|float> $sample
 * @return array<string,float>
 */',
        'startLine' => 403,
        'endLine' => 426,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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
        'startLine' => 435,
        'endLine' => 438,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\GaussianNB',
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