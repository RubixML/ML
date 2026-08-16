<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Initializers/Initializer.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\Initializers\Initializer
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-0bc5aba6839b610c29b82a199f5d8c913975b2f0e3d721332af6050cb44764d3',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\Initializers\\Initializer',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Initializers/Initializer.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\Initializers',
    'name' => 'Rubix\\ML\\NeuralNet\\Initializers\\Initializer',
    'shortName' => 'Initializer',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Initializer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 20,
    'endLine' => 32,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Stringable',
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
            'startLine' => 31,
            'endLine' => 31,
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
            'startLine' => 31,
            'endLine' => 31,
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
 * Initialize a weight matrix W in the dimensions `fanIn` x `fanOut`.
 *
 * @param int<1, max> $fanIn The number of input connections per neuron
 * @param int<1, max> $fanOut The number of output connections per neuron
 * @throws InvalidFanInException Initializer parameter `fanIn` is less than 1
 * @throws InvalidFanOutException Initializer parameter `fanOut` is less than 1
 * @return NDArray The initialized weight matrix of shape [fanOut, fanIn]
 */',
        'startLine' => 31,
        'endLine' => 31,
        'startColumn' => 5,
        'endColumn' => 66,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Initializers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\Initializer',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\Initializer',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Initializers\\Initializer',
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