<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Classifiers/LogitBoost.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Classifiers\LogitBoost
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-08f19a43a05c9d7d106015dc5e22c824bc64f526c812f04124f138dcb77d1d81',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Classifiers/LogitBoost.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Classifiers',
    'name' => 'Rubix\\ML\\Classifiers\\LogitBoost',
    'shortName' => 'LogitBoost',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Logit Boost
 *
 * A stage-wise additive ensemble that uses regression trees to iteratively learn a Logistic Regression model for binary
 * classification problems. Unlike standard Logistic Regression, Logit Boost has the ability to learn a smooth non-linear
 * decision surface by training decision trees to follow the gradient of the cross entropy loss function. In addition,
 * Logit Boost concentrates more effort on classifying samples that it is less certain about.
 *
 * References:
 * [1] J. H. Friedman et al. (2000). Additive Logistic Regression: A Statistical View of Boosting.
 * [2] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
 * [3] J. H. Friedman. (1999). Stochastic Gradient Boosting.
 * [4] Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms: A general analysis with localized complexities.
 * [5] G. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 66,
    'endLine' => 691,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\Probabilistic',
      3 => 'Rubix\\ML\\RanksFeatures',
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
      'COMPATIBLE_BOOSTERS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'name' => 'COMPATIBLE_BOOSTERS',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[\\Rubix\\ML\\Regressors\\RegressionTree::class, \\Rubix\\ML\\Regressors\\ExtraTreeRegressor::class]',
          'attributes' => 
          array (
            'startLine' => 75,
            'endLine' => 78,
            'startTokenPos' => 278,
            'startFilePos' => 2722,
            'endTokenPos' => 290,
            'endFilePos' => 2794,
          ),
        ),
        'docComment' => '/**
 * The class names of the learners that can be used as boosters.
 *
 * @var class-string[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 75,
        'endLine' => 78,
        'startColumn' => 5,
        'endColumn' => 6,
      ),
      'MIN_SUBSAMPLE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'name' => 'MIN_SUBSAMPLE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2',
          'attributes' => 
          array (
            'startLine' => 85,
            'endLine' => 85,
            'startTokenPos' => 303,
            'startFilePos' => 2922,
            'endTokenPos' => 303,
            'endFilePos' => 2922,
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
        'startLine' => 85,
        'endLine' => 85,
        'startColumn' => 5,
        'endColumn' => 38,
      ),
    ),
    'immediateProperties' => 
    array (
      'booster' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
 * The regressor used to fix up error residuals.
 *
 * @var Learner
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 92,
        'endLine' => 92,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 99,
        'endLine' => 99,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
 * The ratio of samples to subsample from the training set for each booster.d
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 106,
        'endLine' => 106,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 113,
        'endLine' => 113,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 120,
        'endLine' => 120,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'name' => 'evalInterval',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The number of epochs to train before evaluating the model with the holdout set.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 127,
        'endLine' => 127,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'window' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
 * The number of epochs without improvement in the validation score to wait before considering an early stop.
 *
 * @var positive-int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 134,
        'endLine' => 134,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 141,
        'endLine' => 141,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 148,
        'endLine' => 148,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'boosters' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'name' => 'boosters',
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
            'startLine' => 155,
            'endLine' => 155,
            'startTokenPos' => 396,
            'startFilePos' => 4476,
            'endTokenPos' => 396,
            'endFilePos' => 4479,
          ),
        ),
        'docComment' => '/**
 * The ensemble of boosters.
 *
 * @var mixed[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 155,
        'endLine' => 155,
        'startColumn' => 5,
        'endColumn' => 38,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'scores' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
            'startLine' => 162,
            'endLine' => 162,
            'startTokenPos' => 410,
            'startFilePos' => 4606,
            'endTokenPos' => 410,
            'endFilePos' => 4609,
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
        'startLine' => 162,
        'endLine' => 162,
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
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
            'startLine' => 169,
            'endLine' => 169,
            'startTokenPos' => 424,
            'startFilePos' => 4740,
            'endTokenPos' => 424,
            'endFilePos' => 4743,
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
        'startLine' => 169,
        'endLine' => 169,
        'startColumn' => 5,
        'endColumn' => 36,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'classes' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'name' => 'classes',
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
            'startLine' => 176,
            'endLine' => 176,
            'startTokenPos' => 438,
            'startFilePos' => 4864,
            'endTokenPos' => 438,
            'endFilePos' => 4867,
          ),
        ),
        'docComment' => '/**
 * The unique class labels.
 *
 * @var list<string>|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 176,
        'endLine' => 176,
        'startColumn' => 5,
        'endColumn' => 37,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'featureCount' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
            'startLine' => 183,
            'endLine' => 183,
            'startTokenPos' => 452,
            'startFilePos' => 5004,
            'endTokenPos' => 452,
            'endFilePos' => 5007,
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
        'startLine' => 183,
        'endLine' => 183,
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
          'booster' => 
          array (
            'name' => 'booster',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 198,
                'endLine' => 198,
                'startTokenPos' => 471,
                'startFilePos' => 5395,
                'endTokenPos' => 471,
                'endFilePos' => 5398,
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
            'startLine' => 198,
            'endLine' => 198,
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
                'startLine' => 199,
                'endLine' => 199,
                'startTokenPos' => 480,
                'startFilePos' => 5423,
                'endTokenPos' => 480,
                'endFilePos' => 5425,
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
                'startLine' => 200,
                'endLine' => 200,
                'startTokenPos' => 489,
                'startFilePos' => 5451,
                'endTokenPos' => 489,
                'endFilePos' => 5453,
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
            'startLine' => 200,
            'endLine' => 200,
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
                'startLine' => 201,
                'endLine' => 201,
                'startTokenPos' => 498,
                'startFilePos' => 5478,
                'endTokenPos' => 498,
                'endFilePos' => 5481,
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
            'startLine' => 201,
            'endLine' => 201,
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
                'startLine' => 202,
                'endLine' => 202,
                'startTokenPos' => 507,
                'startFilePos' => 5511,
                'endTokenPos' => 507,
                'endFilePos' => 5514,
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
            'startLine' => 202,
            'endLine' => 202,
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
                'startLine' => 203,
                'endLine' => 203,
                'startTokenPos' => 516,
                'startFilePos' => 5545,
                'endTokenPos' => 516,
                'endFilePos' => 5545,
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
                'startLine' => 204,
                'endLine' => 204,
                'startTokenPos' => 525,
                'startFilePos' => 5570,
                'endTokenPos' => 525,
                'endFilePos' => 5570,
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
            'startLine' => 204,
            'endLine' => 204,
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
                'startLine' => 205,
                'endLine' => 205,
                'startTokenPos' => 534,
                'startFilePos' => 5598,
                'endTokenPos' => 534,
                'endFilePos' => 5600,
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
            'startLine' => 205,
            'endLine' => 205,
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
                'startLine' => 206,
                'endLine' => 206,
                'startTokenPos' => 544,
                'startFilePos' => 5629,
                'endTokenPos' => 544,
                'endFilePos' => 5632,
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
            'startLine' => 206,
            'endLine' => 206,
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
        'startLine' => 197,
        'endLine' => 261,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 270,
        'endLine' => 273,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 282,
        'endLine' => 285,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 294,
        'endLine' => 307,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 314,
        'endLine' => 319,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 326,
        'endLine' => 339,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 346,
        'endLine' => 349,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 356,
        'endLine' => 359,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
            'startLine' => 366,
            'endLine' => 366,
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
        'startLine' => 366,
        'endLine' => 526,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
            'startLine' => 535,
            'endLine' => 535,
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
 * @return list<string>
 */',
        'startLine' => 535,
        'endLine' => 560,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
            'startLine' => 569,
            'endLine' => 569,
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
        'startLine' => 569,
        'endLine' => 599,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 607,
        'endLine' => 630,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
            'startLine' => 639,
            'endLine' => 639,
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
            'startLine' => 639,
            'endLine' => 639,
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
 * Compute the gradient.
 *
 * @param float $out
 * @param float $target
 * @return float
 */',
        'startLine' => 639,
        'endLine' => 642,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'aliasName' => NULL,
      ),
      'crossEntropy' => 
      array (
        'name' => 'crossEntropy',
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
            'startLine' => 651,
            'endLine' => 651,
            'startColumn' => 37,
            'endColumn' => 46,
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
            'startLine' => 651,
            'endLine' => 651,
            'startColumn' => 49,
            'endColumn' => 61,
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
 * Compute the binary cross entropy loss function.
 *
 * @param float $out
 * @param float $target
 * @return float
 */',
        'startLine' => 651,
        'endLine' => 654,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'aliasName' => NULL,
      ),
      'updateZ' => 
      array (
        'name' => 'updateZ',
        'parameters' => 
        array (
          'z' => 
          array (
            'name' => 'z',
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
            'startLine' => 663,
            'endLine' => 663,
            'startColumn' => 32,
            'endColumn' => 39,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'prevZ' => 
          array (
            'name' => 'prevZ',
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
            'startLine' => 663,
            'endLine' => 663,
            'startColumn' => 42,
            'endColumn' => 53,
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
 * Compute the z signal for an iteration.
 *
 * @param float $z
 * @param float $prevZ
 * @return float
 */',
        'startLine' => 663,
        'endLine' => 666,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
        'startLine' => 673,
        'endLine' => 680,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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
 * @return string
 */',
        'startLine' => 687,
        'endLine' => 690,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Classifiers',
        'declaringClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'implementingClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
        'currentClassName' => 'Rubix\\ML\\Classifiers\\LogitBoost',
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