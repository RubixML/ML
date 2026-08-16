<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Regressors/GradientBoost.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Regressors\GradientBoost
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-0a973492e893e8f281a58a5c7b74da5c1abf2b8772ef493c0fbf626d56dd64f9',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Regressors/GradientBoost.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Regressors',
    'name' => 'Rubix\\ML\\Regressors\\GradientBoost',
    'shortName' => 'GradientBoost',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Gradient Boost
 *
 * Gradient Boost is a stage-wise additive ensemble that uses a Gradient Descent boosting
 * scheme for training  boosters (Decision Trees) to correct the error residuals of a
 * series of *weak* base learners. Stochastic gradient boosting is achieved by varying
 * the ratio of samples to subsample uniformly at random from the training set.
 *
 * References:
 * [1] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
 * [2] J. H. Friedman. (1999). Stochastic Gradient Boosting.
 * [3] Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms: A general analysis
 * with localized complexities.
 * [4] G. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 62,
    'endLine' => 623,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\RanksFeatures',
      3 => 'Rubix\\ML\\Verbose',
      4 => 'Rubix\\ML\\Persistable',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AutotrackRevisions',
      1 => 'Rubix\\ML\\Traits\\LoggerAware',
    ),
    'immediateConstants' => 
    array (
      'COMPATIBLE_BOOSTERS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'COMPATIBLE_BOOSTERS',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[\\Rubix\\ML\\Regressors\\RegressionTree::class, \\Rubix\\ML\\Regressors\\ExtraTreeRegressor::class]',
          'attributes' => 
          array (
            'startLine' => 71,
            'endLine' => 74,
            'startTokenPos' => 253,
            'startFilePos' => 2427,
            'endTokenPos' => 265,
            'endFilePos' => 2499,
          ),
        ),
        'docComment' => '/**
 * The class names of the compatible learners to used as boosters.
 *
 * @var class-string[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 71,
        'endLine' => 74,
        'startColumn' => 5,
        'endColumn' => 6,
      ),
      'MIN_SUBSAMPLE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'MIN_SUBSAMPLE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2',
          'attributes' => 
          array (
            'startLine' => 81,
            'endLine' => 81,
            'startTokenPos' => 278,
            'startFilePos' => 2627,
            'endTokenPos' => 278,
            'endFilePos' => 2627,
          ),
        ),
        'docComment' => '/**
 * The minimum size of each training subset.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 81,
        'endLine' => 81,
        'startColumn' => 5,
        'endColumn' => 38,
      ),
    ),
    'immediateProperties' => 
    array (
      'booster' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'booster',
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
 * The regressor that will fix up the error residuals of the *weak* base learner.
 *
 * @var Learner
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 88,
        'endLine' => 88,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'rate' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'rate',
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
 * The learning rate of the ensemble i.e. the *shrinkage* applied to each step.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 95,
        'endLine' => 95,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'ratio' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
 * The ratio of samples to subsample from the training set for each booster.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 102,
        'endLine' => 102,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'epochs' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
 * The maximum number of training epochs. i.e. the number of times to iterate before terminating.
 *
 * @var int<0,max>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 109,
        'endLine' => 109,
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
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
 * The minimum change in the training loss necessary to continue training.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 116,
        'endLine' => 116,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'evalInterval' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'evalInterval',
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
 * The number of epochs to train before evaluating the model with the holdout set.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 123,
        'endLine' => 123,
        'startColumn' => 5,
        'endColumn' => 32,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'window' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
 * The number of epochs without improvement in the validation score to wait before considering an
 * early stop.
 *
 * @var positive-int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 131,
        'endLine' => 131,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'holdOut' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'holdOut',
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
 * The proportion of training samples to use for validation and progress monitoring.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 138,
        'endLine' => 138,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'metric' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'metric',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\Metric',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The metric used to score the generalization performance of the model during training.
 *
 * @var Metric
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 145,
        'endLine' => 145,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'ensemble' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
            'startLine' => 152,
            'endLine' => 154,
            'startTokenPos' => 372,
            'startFilePos' => 4224,
            'endTokenPos' => 376,
            'endFilePos' => 4241,
          ),
        ),
        'docComment' => '/**
 * An ensemble of weak regressors.
 *
 * @var mixed[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 152,
        'endLine' => 154,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'scores' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'scores',
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
            'startLine' => 161,
            'endLine' => 161,
            'startTokenPos' => 390,
            'startFilePos' => 4368,
            'endTokenPos' => 390,
            'endFilePos' => 4371,
          ),
        ),
        'docComment' => '/**
 * The validation scores at each epoch.
 *
 * @var float[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 161,
        'endLine' => 161,
        'startColumn' => 5,
        'endColumn' => 36,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'losses' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
            'startLine' => 168,
            'endLine' => 168,
            'startTokenPos' => 404,
            'startFilePos' => 4502,
            'endTokenPos' => 404,
            'endFilePos' => 4505,
          ),
        ),
        'docComment' => '/**
 * The average training loss at each epoch.
 *
 * @var float[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 168,
        'endLine' => 168,
        'startColumn' => 5,
        'endColumn' => 36,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'featureCount' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
            'startLine' => 175,
            'endLine' => 175,
            'startTokenPos' => 418,
            'startFilePos' => 4642,
            'endTokenPos' => 418,
            'endFilePos' => 4645,
          ),
        ),
        'docComment' => '/**
 * The dimensionality of the training set.
 *
 * @var int<0,max>|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 175,
        'endLine' => 175,
        'startColumn' => 5,
        'endColumn' => 40,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'mu' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'name' => 'mu',
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
            'startLine' => 182,
            'endLine' => 182,
            'startTokenPos' => 432,
            'startFilePos' => 4773,
            'endTokenPos' => 432,
            'endFilePos' => 4776,
          ),
        ),
        'docComment' => '/**
 * The mean of the labels of the training set.
 *
 * @var float|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 182,
        'endLine' => 182,
        'startColumn' => 5,
        'endColumn' => 32,
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
          'booster' => 
          array (
            'name' => 'booster',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 197,
                'endLine' => 197,
                'startTokenPos' => 451,
                'startFilePos' => 5164,
                'endTokenPos' => 451,
                'endFilePos' => 5167,
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
                      'name' => 'Rubix\\ML\\Learner',
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
            'startLine' => 197,
            'endLine' => 197,
            'startColumn' => 9,
            'endColumn' => 32,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'rate' => 
          array (
            'name' => 'rate',
            'default' => 
            array (
              'code' => '0.1',
              'attributes' => 
              array (
                'startLine' => 198,
                'endLine' => 198,
                'startTokenPos' => 460,
                'startFilePos' => 5192,
                'endTokenPos' => 460,
                'endFilePos' => 5194,
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
            'startLine' => 198,
            'endLine' => 198,
            'startColumn' => 9,
            'endColumn' => 25,
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
                'startLine' => 199,
                'endLine' => 199,
                'startTokenPos' => 469,
                'startFilePos' => 5220,
                'endTokenPos' => 469,
                'endFilePos' => 5222,
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
            'startLine' => 199,
            'endLine' => 199,
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 2,
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
                'startLine' => 200,
                'endLine' => 200,
                'startTokenPos' => 478,
                'startFilePos' => 5247,
                'endTokenPos' => 478,
                'endFilePos' => 5250,
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
            'startLine' => 200,
            'endLine' => 200,
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 3,
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
                'startLine' => 201,
                'endLine' => 201,
                'startTokenPos' => 487,
                'startFilePos' => 5280,
                'endTokenPos' => 487,
                'endFilePos' => 5283,
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
            'startLine' => 201,
            'endLine' => 201,
            'startColumn' => 9,
            'endColumn' => 31,
            'parameterIndex' => 4,
            'isOptional' => true,
          ),
          'evalInterval' => 
          array (
            'name' => 'evalInterval',
            'default' => 
            array (
              'code' => '3',
              'attributes' => 
              array (
                'startLine' => 202,
                'endLine' => 202,
                'startTokenPos' => 496,
                'startFilePos' => 5314,
                'endTokenPos' => 496,
                'endFilePos' => 5314,
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
            'startLine' => 202,
            'endLine' => 202,
            'startColumn' => 9,
            'endColumn' => 29,
            'parameterIndex' => 5,
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
                'startLine' => 203,
                'endLine' => 203,
                'startTokenPos' => 505,
                'startFilePos' => 5339,
                'endTokenPos' => 505,
                'endFilePos' => 5339,
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
            'startLine' => 203,
            'endLine' => 203,
            'startColumn' => 9,
            'endColumn' => 23,
            'parameterIndex' => 6,
            'isOptional' => true,
          ),
          'holdOut' => 
          array (
            'name' => 'holdOut',
            'default' => 
            array (
              'code' => '0.1',
              'attributes' => 
              array (
                'startLine' => 204,
                'endLine' => 204,
                'startTokenPos' => 514,
                'startFilePos' => 5367,
                'endTokenPos' => 514,
                'endFilePos' => 5369,
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
            'startLine' => 204,
            'endLine' => 204,
            'startColumn' => 9,
            'endColumn' => 28,
            'parameterIndex' => 7,
            'isOptional' => true,
          ),
          'metric' => 
          array (
            'name' => 'metric',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 205,
                'endLine' => 205,
                'startTokenPos' => 524,
                'startFilePos' => 5398,
                'endTokenPos' => 524,
                'endFilePos' => 5401,
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
                      'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\Metric',
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
            'startLine' => 205,
            'endLine' => 205,
            'startColumn' => 9,
            'endColumn' => 30,
            'parameterIndex' => 8,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param Learner|null $booster
 * @param float $rate
 * @param float $ratio
 * @param int $epochs
 * @param float $minChange
 * @param int $evalInterval
 * @param int $window
 * @param float $holdOut
 * @param Metric|null $metric
 * @throws InvalidArgumentException
 */',
        'startLine' => 196,
        'endLine' => 260,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
        'startLine' => 269,
        'endLine' => 272,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
        'startLine' => 281,
        'endLine' => 284,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
        'startLine' => 293,
        'endLine' => 306,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
        'startLine' => 313,
        'endLine' => 316,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
        'startLine' => 323,
        'endLine' => 336,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'aliasName' => NULL,
      ),
      'scores' => 
      array (
        'name' => 'scores',
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
 * Return the validation scores at each epoch from the last training session.
 *
 * @return float[]|null
 */',
        'startLine' => 343,
        'endLine' => 346,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
        'startLine' => 353,
        'endLine' => 356,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Train the estimator with a dataset.
 *
 * @param Labeled $dataset
 */',
        'startLine' => 363,
        'endLine' => 503,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
            'startLine' => 512,
            'endLine' => 512,
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
 * Make a prediction from a dataset.
 *
 * @param Dataset $dataset
 * @throws RuntimeException
 * @return list<int|float>
 */',
        'startLine' => 512,
        'endLine' => 529,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'aliasName' => NULL,
      ),
      'featureImportances' => 
      array (
        'name' => 'featureImportances',
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
 * Return the importance scores of each feature column of the training set.
 *
 * @throws RuntimeException
 * @return float[]
 */',
        'startLine' => 537,
        'endLine' => 560,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'aliasName' => NULL,
      ),
      'updateOut' => 
      array (
        'name' => 'updateOut',
        'parameters' => 
        array (
          'prediction' => 
          array (
            'name' => 'prediction',
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
            'startLine' => 569,
            'endLine' => 569,
            'startColumn' => 34,
            'endColumn' => 50,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'out' => 
          array (
            'name' => 'out',
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
            'startLine' => 569,
            'endLine' => 569,
            'startColumn' => 53,
            'endColumn' => 62,
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
 * Compute the output for an iteration.
 *
 * @param float $prediction
 * @param float $out
 * @return float
 */',
        'startLine' => 569,
        'endLine' => 572,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'aliasName' => NULL,
      ),
      'gradient' => 
      array (
        'name' => 'gradient',
        'parameters' => 
        array (
          'out' => 
          array (
            'name' => 'out',
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
            'startLine' => 581,
            'endLine' => 581,
            'startColumn' => 33,
            'endColumn' => 42,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'target' => 
          array (
            'name' => 'target',
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
            'startLine' => 581,
            'endLine' => 581,
            'startColumn' => 45,
            'endColumn' => 57,
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
 * Compute the gradient for a single sample.
 *
 * @param float $out
 * @param float $target
 * @return float
 */',
        'startLine' => 581,
        'endLine' => 584,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'aliasName' => NULL,
      ),
      'l2Loss' => 
      array (
        'name' => 'l2Loss',
        'parameters' => 
        array (
          'loss' => 
          array (
            'name' => 'loss',
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
            'startLine' => 593,
            'endLine' => 593,
            'startColumn' => 31,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'derivative' => 
          array (
            'name' => 'derivative',
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
            'startLine' => 593,
            'endLine' => 593,
            'startColumn' => 44,
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
 * Compute the cross entropy loss function.
 *
 * @param float $loss
 * @param float $derivative
 * @return float
 */',
        'startLine' => 593,
        'endLine' => 596,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
        'startLine' => 603,
        'endLine' => 610,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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
        'startLine' => 619,
        'endLine' => 622,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
        'currentClassName' => 'Rubix\\ML\\Regressors\\GradientBoost',
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