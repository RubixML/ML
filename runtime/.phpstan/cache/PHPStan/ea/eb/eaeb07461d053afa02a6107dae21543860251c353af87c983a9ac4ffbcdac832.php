<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Optimizers/Cyclical.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\Optimizers\Cyclical
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-5ca4fea80eed732bb9f150595f756bfa0a0edd0086ad0ac3732d58914686241a',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Optimizers/Cyclical.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
    'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
    'shortName' => 'Cyclical',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Cyclical
 *
 * The Cyclical optimizer uses a global learning rate that cycles between the
 * lower and upper bound over a designated period while also decaying the
 * upper bound by the decay coefficient at each step. Cyclical learning rates
 * have been shown to help escape bad local minima and saddle points thus
 * achieving lower training loss.
 *
 * References:
 * [1] L. N. Smith. (2017). Cyclical Learning Rates for Training Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 30,
    'endLine' => 173,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\NeuralNet\\Optimizers\\Optimizer',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'lower' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'name' => 'lower',
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
 * The lower bound on the learning rate.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'upper' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'name' => 'upper',
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
 * The upper bound on the learning rate.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 44,
        'endLine' => 44,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'range' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'name' => 'range',
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
 * The range of the learning rate.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 51,
        'endLine' => 51,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'losses' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'name' => 'losses',
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
 * The number of steps in every cycle.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 58,
        'endLine' => 58,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'decay' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'name' => 'decay',
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
 * The exponential scaling factor applied to each step as decay.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 65,
        'endLine' => 65,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      't' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'name' => 't',
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
            'startLine' => 72,
            'endLine' => 72,
            'startTokenPos' => 109,
            'startFilePos' => 1643,
            'endTokenPos' => 109,
            'endFilePos' => 1643,
          ),
        ),
        'docComment' => '/**
 * The number of steps taken so far.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 72,
        'endLine' => 72,
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
          'lower' => 
          array (
            'name' => 'lower',
            'default' => 
            array (
              'code' => '0.001',
              'attributes' => 
              array (
                'startLine' => 82,
                'endLine' => 82,
                'startTokenPos' => 127,
                'startFilePos' => 1866,
                'endTokenPos' => 127,
                'endFilePos' => 1870,
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
            'startLine' => 82,
            'endLine' => 82,
            'startColumn' => 9,
            'endColumn' => 28,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'upper' => 
          array (
            'name' => 'upper',
            'default' => 
            array (
              'code' => '0.006',
              'attributes' => 
              array (
                'startLine' => 83,
                'endLine' => 83,
                'startTokenPos' => 136,
                'startFilePos' => 1896,
                'endTokenPos' => 136,
                'endFilePos' => 1900,
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
            'startLine' => 83,
            'endLine' => 83,
            'startColumn' => 9,
            'endColumn' => 28,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'losses' => 
          array (
            'name' => 'losses',
            'default' => 
            array (
              'code' => '2000',
              'attributes' => 
              array (
                'startLine' => 84,
                'endLine' => 84,
                'startTokenPos' => 145,
                'startFilePos' => 1925,
                'endTokenPos' => 145,
                'endFilePos' => 1928,
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
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
          'decay' => 
          array (
            'name' => 'decay',
            'default' => 
            array (
              'code' => '0.9999400000000001',
              'attributes' => 
              array (
                'startLine' => 85,
                'endLine' => 85,
                'startTokenPos' => 154,
                'startFilePos' => 1954,
                'endTokenPos' => 154,
                'endFilePos' => 1960,
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
            'startLine' => 85,
            'endLine' => 85,
            'startColumn' => 9,
            'endColumn' => 30,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param float $lower
 * @param float $upper
 * @param int $losses
 * @param float $decay
 * @throws InvalidArgumentException
 */',
        'startLine' => 81,
        'endLine' => 121,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'aliasName' => NULL,
      ),
      'step' => 
      array (
        'name' => 'step',
        'parameters' => 
        array (
          'param' => 
          array (
            'name' => 'param',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\NeuralNet\\Parameter',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 146,
            'endLine' => 146,
            'startColumn' => 26,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'gradient' => 
          array (
            'name' => 'gradient',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'NDArray',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 146,
            'endLine' => 146,
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
            'name' => 'NDArray',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Take a step of gradient descent for a given parameter.
 *
 * Cyclical learning rate schedule (per-step, element-wise update):
 *   - Cycle index:           cycle = floor(1 + t / (2 · losses))
 *   - Triangular position:   x     = | t / losses − 2 · cycle + 1 |
 *   - Exponential decay:     scale = decay^t
 *   - Learning rate at t:    η_t   = lower + (upper − lower) · max(0, 1 − x) · scale
 *   - Returned step:         Δθ_t  = η_t · g_t
 *
 * where:
 *   - t is the current step counter (incremented after computing η_t),
 *   - losses is the number of steps per cycle,
 *   - lower and upper are the learning rate bounds,
 *   - decay is the multiplicative decay applied each step,
 *   - g_t is the current gradient.
 *
 * @internal
 *
 * @param Parameter $param
 * @param NDArray $gradient
 * @return NDArray
 */',
        'startLine' => 146,
        'endLine' => 159,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
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
        'startLine' => 168,
        'endLine' => 172,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Cyclical',
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