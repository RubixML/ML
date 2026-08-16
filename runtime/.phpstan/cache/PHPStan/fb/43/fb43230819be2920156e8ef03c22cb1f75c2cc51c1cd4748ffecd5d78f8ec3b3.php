<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Initializers/HeUniform.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\Initializers\HeUniform
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-a73f97e9333cecdf3e4eca70ea78dd8e61c2f107ad1e6b84335fdecbfe9bf898',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Initializers/HeUniform.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\Initializers',
    'name' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
    'shortName' => 'HeUniform',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * He Uniform
 *
 * The He initializer was designed for hidden layers that feed into rectified
 * linear layers such ReLU, Leaky ReLU, ELU, and SELU. It draws from a uniform
 * distribution with limits +/- sqrt(6 / fanOut).
 *
 * References:
 * [1] K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
 * Performance on ImageNet Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 29,
    'endLine' => 60,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\AbstractInitializer',
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => NULL,
        'startLine' => 31,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Initializers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
        'aliasName' => NULL,
      ),
      'initialize' => 
      array (
        'name' => 'initialize',
        'parameters' => 
        array (
          'fanIn' => 
          array (
            'name' => 'fanIn',
            'default' => NULL,
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
            'startLine' => 42,
            'endLine' => 42,
            'startColumn' => 32,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'fanOut' => 
          array (
            'name' => 'fanOut',
            'default' => NULL,
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
            'startLine' => 42,
            'endLine' => 42,
            'startColumn' => 44,
            'endColumn' => 54,
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
 * @inheritdoc
 */',
        'startLine' => 42,
        'endLine' => 49,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Initializers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
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
 * Return the string representation of the initializer.
 *
 * @return string String representation
 */',
        'startLine' => 56,
        'endLine' => 59,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Initializers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\HeUniform',
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