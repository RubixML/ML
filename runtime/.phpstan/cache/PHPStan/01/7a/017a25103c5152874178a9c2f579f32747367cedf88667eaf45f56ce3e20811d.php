<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Optimizers/Momentum.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\Optimizers\Momentum
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-c7d305a789dcef4f17c0781409bbe8bb6722e3a63f03d991e172ee8cb90b87d2',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Optimizers/Momentum.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
    'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
    'shortName' => 'Momentum',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Momentum
 *
 * Momentum adds velocity to each step until exhausted. It does so by accumulating momentum from past updates and adding
 * a factor of the previous velocity to the current step.
 *
 * References:
 * [1] D. E. Rumelhart et al. (1988). Learning representations by back-propagating errors.
 * [2] I. Sutskever et al. (2013). On the importance of initialization and momentum in deep learning.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 32,
    'endLine' => 170,
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
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
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
        'startLine' => 39,
        'endLine' => 39,
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
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
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
 * The rate at which the momentum force decays.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 46,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'lookahead' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'name' => 'lookahead',
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
 * Should we employ Nesterov\'s lookahead (NAG) when updating the parameters?
 *
 * @var bool
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 53,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'cache' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
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
            'startLine' => 60,
            'endLine' => 62,
            'startTokenPos' => 111,
            'startFilePos' => 1538,
            'endTokenPos' => 115,
            'endFilePos' => 1555,
          ),
        ),
        'docComment' => '/**
 * The parameter cache of velocity NDArrays.
 *
 * @var NDArray[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 60,
        'endLine' => 62,
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
                'startLine' => 70,
                'endLine' => 70,
                'startTokenPos' => 132,
                'startFilePos' => 1744,
                'endTokenPos' => 132,
                'endFilePos' => 1748,
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
            'startLine' => 70,
            'endLine' => 70,
            'startColumn' => 33,
            'endColumn' => 51,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'decay' => 
          array (
            'name' => 'decay',
            'default' => 
            array (
              'code' => '0.1',
              'attributes' => 
              array (
                'startLine' => 70,
                'endLine' => 70,
                'startTokenPos' => 141,
                'startFilePos' => 1766,
                'endTokenPos' => 141,
                'endFilePos' => 1768,
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
            'startLine' => 70,
            'endLine' => 70,
            'startColumn' => 54,
            'endColumn' => 71,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'lookahead' => 
          array (
            'name' => 'lookahead',
            'default' => 
            array (
              'code' => 'false',
              'attributes' => 
              array (
                'startLine' => 70,
                'endLine' => 70,
                'startTokenPos' => 150,
                'startFilePos' => 1789,
                'endTokenPos' => 150,
                'endFilePos' => 1793,
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
            'startLine' => 70,
            'endLine' => 70,
            'startColumn' => 74,
            'endColumn' => 96,
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
 * @param float $decay
 * @param bool $lookahead
 * @throws InvalidArgumentException
 */',
        'startLine' => 70,
        'endLine' => 92,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
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
            'startLine' => 102,
            'endLine' => 102,
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
 * Warm the cache.
 *
 * @internal
 *
 * @param Parameter $param
 * @throws RuntimeException
 */',
        'startLine' => 102,
        'endLine' => 111,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
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
            'startLine' => 135,
            'endLine' => 135,
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
            'startLine' => 135,
            'endLine' => 135,
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
 * Mathematical formulation (per-parameter element):
 * - Velocity update: v_t = β · v_{t-1} + η · g_t
 *   where β = 1 − decay and η = rate, and g_t is the current gradient.
 * - Returned step (the amount added to the parameter by the trainer): Δθ_t = v_t
 *
 * Nesterov lookahead (when lookahead = true):
 * - We apply the same velocity update a second time to approximate NAG:
 *   v_t ← β · v_t + η · g_t
 *
 * Notes:
 * - This method updates and caches the velocity tensor per Parameter id.
 * - The actual parameter update is performed by the training loop using the returned velocity.
 *
 * @internal
 *
 * @param Parameter $param
 * @param NDArray $gradient
 * @return NDArray
 */',
        'startLine' => 135,
        'endLine' => 156,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
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
        'startLine' => 165,
        'endLine' => 169,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Momentum',
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