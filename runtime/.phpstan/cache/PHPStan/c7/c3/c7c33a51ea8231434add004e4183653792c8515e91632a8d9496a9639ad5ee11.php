<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Optimizers/StepDecay.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\Optimizers\StepDecay
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-b71d13b4d7b2908b319161160513984366bbf8699b364393b08f67648864300d',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Optimizers/StepDecay.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
    'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
    'shortName' => 'StepDecay',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Step Decay
 *
 * A linear learning rate scheduler that reduces the learning rate by a factor
 * of the decay parameter whenever it reaches a new *floor*. The number of
 * steps needed to reach a new floor is defined by the *steps* parameter.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 25,
    'endLine' => 134,
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
      'rate' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
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
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'losses' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
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
 * The size of every floor in steps. i.e. the number of steps to take before applying another factor of decay.
 *
 * @var int
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
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
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
 * The factor to decrease the learning rate by over a period of k steps.
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
      'steps' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'name' => 'steps',
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
            'startLine' => 53,
            'endLine' => 53,
            'startTokenPos' => 91,
            'startFilePos' => 1311,
            'endTokenPos' => 91,
            'endFilePos' => 1311,
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
        'startLine' => 53,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 29,
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
              'code' => '0.01',
              'attributes' => 
              array (
                'startLine' => 61,
                'endLine' => 61,
                'startTokenPos' => 108,
                'startFilePos' => 1496,
                'endTokenPos' => 108,
                'endFilePos' => 1499,
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
            'startLine' => 61,
            'endLine' => 61,
            'startColumn' => 33,
            'endColumn' => 50,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'losses' => 
          array (
            'name' => 'losses',
            'default' => 
            array (
              'code' => '100',
              'attributes' => 
              array (
                'startLine' => 61,
                'endLine' => 61,
                'startTokenPos' => 117,
                'startFilePos' => 1516,
                'endTokenPos' => 117,
                'endFilePos' => 1518,
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
            'startLine' => 61,
            'endLine' => 61,
            'startColumn' => 53,
            'endColumn' => 69,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'decay' => 
          array (
            'name' => 'decay',
            'default' => 
            array (
              'code' => '0.001',
              'attributes' => 
              array (
                'startLine' => 61,
                'endLine' => 61,
                'startTokenPos' => 126,
                'startFilePos' => 1536,
                'endTokenPos' => 126,
                'endFilePos' => 1539,
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
            'startLine' => 61,
            'endLine' => 61,
            'startColumn' => 72,
            'endColumn' => 90,
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
 * @param int $losses
 * @param float $decay
 * @throws InvalidArgumentException
 */',
        'startLine' => 61,
        'endLine' => 89,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
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
            'startLine' => 112,
            'endLine' => 112,
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
            'startLine' => 112,
            'endLine' => 112,
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
 * Step Decay update (element-wise):
 *   floor = ⌊t / k⌋
 *   η_t = η₀ / (1 + floor · λ)
 *   Δθ_t = η_t · g_t
 *
 * where:
 *   - t is the current step number,
 *   - k is the number of steps per floor,
 *   - η₀ is the initial learning rate,
 *   - λ is the decay factor,
 *   - g_t is the current gradient.
 *
 * @internal
 *
 * @param Parameter $param
 * @param NDArray $gradient
 * @return NDArray
 */',
        'startLine' => 112,
        'endLine' => 121,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
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
        'startLine' => 130,
        'endLine' => 133,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Optimizers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Optimizers\\StepDecay',
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