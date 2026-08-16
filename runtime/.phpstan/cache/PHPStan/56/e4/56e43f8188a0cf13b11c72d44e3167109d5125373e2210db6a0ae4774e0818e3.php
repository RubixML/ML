<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Regressors/MLPRegressor.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Regressors\MLPRegressor
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-ff49945e47fdf8e2fe762cc69f0e93bf54ebe546d6b5ddf9166d7e9680e7d6f1',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Regressors/MLPRegressor.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Regressors',
    'name' => 'Rubix\\ML\\Regressors\\MLPRegressor',
    'shortName' => 'MLPRegressor',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * MLP Regressor
 *
 * A multilayer feed forward neural network with a continuous output layer suitable for
 * regression problems. Like the Multilayer Perceptron classifier, the MLP Regressor is
 * able to handle complex non-linear regression problems by forming higher-order
 * representations of the input features using intermediate hidden layers.
 *
 * References:
 * [1] G. E. Hinton. (1989). Connectionist learning procedures.
 * [2] L. Prechelt. (1997). Early Stopping - but when?
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 63,
    'endLine' => 558,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
      1 => 'Rubix\\ML\\Learner',
      2 => 'Rubix\\ML\\Online',
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
    ),
    'immediateProperties' => 
    array (
      'hiddenLayers' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'name' => 'hiddenLayers',
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
            'startLine' => 72,
            'endLine' => 74,
            'startTokenPos' => 257,
            'startFilePos' => 2466,
            'endTokenPos' => 261,
            'endFilePos' => 2483,
          ),
        ),
        'docComment' => '/**
 * An array composing the user-specified hidden layers of the network in order.
 *
 * @var Hidden[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 72,
        'endLine' => 74,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'batchSize' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
 * The number of training samples to process at a time.
 *
 * @var positive-int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 81,
        'endLine' => 81,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'optimizer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'name' => 'optimizer',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Optimizer',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The gradient descent optimizer used to update the network parameters.
 *
 * @var Optimizer
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 88,
        'endLine' => 88,
        'startColumn' => 5,
        'endColumn' => 35,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'epochs' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
      'minChange' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 102,
        'endLine' => 102,
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
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 109,
        'endLine' => 109,
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
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 116,
        'endLine' => 116,
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
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 123,
        'endLine' => 123,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'costFn' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'name' => 'costFn',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\RegressionLoss',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The function that computes the loss associated with an erroneous activation during training.
 *
 * @var RegressionLoss
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 130,
        'endLine' => 130,
        'startColumn' => 5,
        'endColumn' => 37,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'metric' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 137,
        'endLine' => 137,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'network' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'name' => 'network',
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
                  'name' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
            'startLine' => 144,
            'endLine' => 144,
            'startTokenPos' => 356,
            'startFilePos' => 4117,
            'endTokenPos' => 356,
            'endFilePos' => 4120,
          ),
        ),
        'docComment' => '/**
 * The underlying neural network instance.
 *
 * @var FeedForward|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 144,
        'endLine' => 144,
        'startColumn' => 5,
        'endColumn' => 43,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'scores' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
            'startLine' => 151,
            'endLine' => 151,
            'startTokenPos' => 370,
            'startFilePos' => 4278,
            'endTokenPos' => 370,
            'endFilePos' => 4281,
          ),
        ),
        'docComment' => '/**
 * The validation scores at each epoch from the last training session.
 *
 * @var float[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 151,
        'endLine' => 151,
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
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
            'startLine' => 158,
            'endLine' => 158,
            'startTokenPos' => 384,
            'startFilePos' => 4426,
            'endTokenPos' => 384,
            'endFilePos' => 4429,
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
        'startLine' => 158,
        'endLine' => 158,
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
          'hiddenLayers' => 
          array (
            'name' => 'hiddenLayers',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 173,
                'endLine' => 173,
                'startTokenPos' => 402,
                'startFilePos' => 4841,
                'endTokenPos' => 403,
                'endFilePos' => 4842,
              ),
            ),
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
            'startLine' => 173,
            'endLine' => 173,
            'startColumn' => 9,
            'endColumn' => 32,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'batchSize' => 
          array (
            'name' => 'batchSize',
            'default' => 
            array (
              'code' => '128',
              'attributes' => 
              array (
                'startLine' => 174,
                'endLine' => 174,
                'startTokenPos' => 412,
                'startFilePos' => 4870,
                'endTokenPos' => 412,
                'endFilePos' => 4872,
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
            'startLine' => 174,
            'endLine' => 174,
            'startColumn' => 9,
            'endColumn' => 28,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'optimizer' => 
          array (
            'name' => 'optimizer',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 175,
                'endLine' => 175,
                'startTokenPos' => 422,
                'startFilePos' => 4907,
                'endTokenPos' => 422,
                'endFilePos' => 4910,
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
                      'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Optimizer',
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
            'startLine' => 175,
            'endLine' => 175,
            'startColumn' => 9,
            'endColumn' => 36,
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
                'startLine' => 176,
                'endLine' => 176,
                'startTokenPos' => 431,
                'startFilePos' => 4935,
                'endTokenPos' => 431,
                'endFilePos' => 4938,
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
            'startLine' => 176,
            'endLine' => 176,
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
                'startLine' => 177,
                'endLine' => 177,
                'startTokenPos' => 440,
                'startFilePos' => 4968,
                'endTokenPos' => 440,
                'endFilePos' => 4971,
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
            'startLine' => 177,
            'endLine' => 177,
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
                'startLine' => 178,
                'endLine' => 178,
                'startTokenPos' => 449,
                'startFilePos' => 5002,
                'endTokenPos' => 449,
                'endFilePos' => 5002,
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
            'startLine' => 178,
            'endLine' => 178,
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
                'startLine' => 179,
                'endLine' => 179,
                'startTokenPos' => 458,
                'startFilePos' => 5027,
                'endTokenPos' => 458,
                'endFilePos' => 5027,
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
            'startLine' => 179,
            'endLine' => 179,
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
                'startLine' => 180,
                'endLine' => 180,
                'startTokenPos' => 467,
                'startFilePos' => 5055,
                'endTokenPos' => 467,
                'endFilePos' => 5057,
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
            'startLine' => 180,
            'endLine' => 180,
            'startColumn' => 9,
            'endColumn' => 28,
            'parameterIndex' => 7,
            'isOptional' => true,
          ),
          'costFn' => 
          array (
            'name' => 'costFn',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 181,
                'endLine' => 181,
                'startTokenPos' => 477,
                'startFilePos' => 5094,
                'endTokenPos' => 477,
                'endFilePos' => 5097,
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
                      'name' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\RegressionLoss',
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
            'startLine' => 181,
            'endLine' => 181,
            'startColumn' => 9,
            'endColumn' => 38,
            'parameterIndex' => 8,
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
                'startLine' => 182,
                'endLine' => 182,
                'startTokenPos' => 487,
                'startFilePos' => 5126,
                'endTokenPos' => 487,
                'endFilePos' => 5129,
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
            'startLine' => 182,
            'endLine' => 182,
            'startColumn' => 9,
            'endColumn' => 30,
            'parameterIndex' => 9,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param list<mixed> $hiddenLayers
 * @param int $batchSize
 * @param Optimizer|null $optimizer
 * @param int $epochs
 * @param float $minChange
 * @param int $evalInterval
 * @param int $window
 * @param float $holdOut
 * @param RegressionLoss|null $costFn
 * @param Metric|null $metric
 */',
        'startLine' => 172,
        'endLine' => 235,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 244,
        'endLine' => 247,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 256,
        'endLine' => 261,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 270,
        'endLine' => 284,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 291,
        'endLine' => 294,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 301,
        'endLine' => 314,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
 * Return the validation score at each epoch.
 *
 * @return float[]|null
 */',
        'startLine' => 321,
        'endLine' => 324,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
 * Return the training loss at each epoch.
 *
 * @return float[]|null
 */',
        'startLine' => 331,
        'endLine' => 334,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'aliasName' => NULL,
      ),
      'network' => 
      array (
        'name' => 'network',
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
                  'name' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the underlying neural network instance or null if not trained.
 *
 * @return FeedForward|null
 */',
        'startLine' => 341,
        'endLine' => 344,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
            'startLine' => 351,
            'endLine' => 351,
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
        'startLine' => 351,
        'endLine' => 369,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
            'startLine' => 377,
            'endLine' => 377,
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
 * Train the network using mini-batch gradient descent with backpropagation.
 *
 * @param Labeled $dataset
 * @throws RuntimeException
 */',
        'startLine' => 377,
        'endLine' => 495,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
            'startLine' => 505,
            'endLine' => 505,
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
 * Feed a sample through the network and make a prediction based on the
 * activation of the output neuron.
 *
 * @param Dataset $dataset
 * @throws RuntimeException
 * @return list<int|float>
 */',
        'startLine' => 505,
        'endLine' => 516,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'aliasName' => NULL,
      ),
      'exportGraphviz' => 
      array (
        'name' => 'exportGraphviz',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Encoding',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Export the network architecture as a graph in dot format.
 *
 * @throws RuntimeException
 * @return Encoding
 */',
        'startLine' => 524,
        'endLine' => 531,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 538,
        'endLine' => 545,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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
        'startLine' => 554,
        'endLine' => 557,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Regressors',
        'declaringClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'implementingClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
        'currentClassName' => 'Rubix\\ML\\Regressors\\MLPRegressor',
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