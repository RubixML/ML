<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Optimizers/Adam.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\Optimizers\Adam
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-07b70b86e8b6abb19cbad14115490c8c1b32b46dfc4ad350fe1dc44de52b23d4',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Optimizers/Adam.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
    'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
    'shortName' => 'Adam',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Adam
 *
 * Short for *Adaptive Moment Estimation*, the Adam Optimizer combines both
 * Momentum and RMS prop to achieve a balance of velocity and stability. In
 * addition to storing an exponentially decaying average of past squared
 * gradients like RMSprop, Adam also keeps an exponentially decaying average
 * of past gradients, similar to Momentum. Whereas Momentum can be seen as a
 * ball running down a slope, Adam behaves like a heavy ball with friction.
 *
 * References:
 * [1] D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 37,
    'endLine' => 188,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\NeuralNet\\Optimizers\\Optimizer',
      1 => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adaptive',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'rate' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
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
 * The learning rate that controls the global step size.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 44,
        'endLine' => 44,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'momentumDecay' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'name' => 'momentumDecay',
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
 * The momentum decay rate.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 51,
        'endLine' => 51,
        'startColumn' => 5,
        'endColumn' => 35,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'normDecay' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'name' => 'normDecay',
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
 * The decay rate of the previous norms.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 58,
        'endLine' => 58,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'cache' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'name' => 'cache',
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
            'startLine' => 65,
            'endLine' => 67,
            'startTokenPos' => 120,
            'startFilePos' => 1714,
            'endTokenPos' => 124,
            'endFilePos' => 1754,
          ),
        ),
        'docComment' => '/**
 * The parameter cache of running velocity and squared gradients.
 *
 * @var array{0: NDArray, 1: NDArray}[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 65,
        'endLine' => 67,
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
          'rate' => 
          array (
            'name' => 'rate',
            'default' => 
            array (
              'code' => '0.001',
              'attributes' => 
              array (
                'startLine' => 75,
                'endLine' => 75,
                'startTokenPos' => 141,
                'startFilePos' => 1952,
                'endTokenPos' => 141,
                'endFilePos' => 1956,
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
            'startLine' => 75,
            'endLine' => 75,
            'startColumn' => 33,
            'endColumn' => 51,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'momentumDecay' => 
          array (
            'name' => 'momentumDecay',
            'default' => 
            array (
              'code' => '0.1',
              'attributes' => 
              array (
                'startLine' => 75,
                'endLine' => 75,
                'startTokenPos' => 150,
                'startFilePos' => 1982,
                'endTokenPos' => 150,
                'endFilePos' => 1984,
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
            'startLine' => 75,
            'endLine' => 75,
            'startColumn' => 54,
            'endColumn' => 79,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'normDecay' => 
          array (
            'name' => 'normDecay',
            'default' => 
            array (
              'code' => '0.001',
              'attributes' => 
              array (
                'startLine' => 75,
                'endLine' => 75,
                'startTokenPos' => 159,
                'startFilePos' => 2006,
                'endTokenPos' => 159,
                'endFilePos' => 2010,
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
            'startLine' => 75,
            'endLine' => 75,
            'startColumn' => 82,
            'endColumn' => 105,
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
 * @param float $rate
 * @param float $momentumDecay
 * @param float $normDecay
 * @throws InvalidArgumentException
 */',
        'startLine' => 75,
        'endLine' => 103,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'aliasName' => NULL,
      ),
      'warm' => 
      array (
        'name' => 'warm',
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
            'startLine' => 113,
            'endLine' => 113,
            'startColumn' => 26,
            'endColumn' => 41,
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
 * Warm the parameter cache.
 *
 * @internal
 *
 * @param Parameter $param
 * @throws RuntimeException
 */',
        'startLine' => 113,
        'endLine' => 125,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
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
            'startLine' => 147,
            'endLine' => 147,
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
            'startLine' => 147,
            'endLine' => 147,
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
 * Adam update (element-wise):
 *   v_t = v_{t-1} + β1 · (g_t − v_{t-1})        // exponential moving average of gradients
 *   n_t = n_{t-1} + β2 · (g_t^2 − n_{t-1})      // exponential moving average of squared gradients
 *   Δθ_t = η · v_t / max(√n_t, ε)
 *
 * where:
 *   - g_t is the current gradient,
 *   - v_t is the running average of gradients ("velocity"), β1 = momentumDecay,
 *   - n_t is the running average of squared gradients ("norm"), β2 = normDecay,
 *   - η is the learning rate (rate), ε is a small constant to avoid division by zero (implemented by clipping √n_t to [ε, +∞)).
 *
 * @internal
 *
 * @param Parameter $param
 * @param NDArray $gradient
 * @return NDArray
 */',
        'startLine' => 147,
        'endLine' => 174,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
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
        'startLine' => 183,
        'endLine' => 187,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Adam',
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