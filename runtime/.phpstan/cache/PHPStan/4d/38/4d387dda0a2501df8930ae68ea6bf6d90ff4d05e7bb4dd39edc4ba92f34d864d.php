<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Network.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\Network
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-ff52298ed3b879b7ffa76f7689df07aeb014df0d13445f6f8e9c78584e4400a0',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\Network',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Network.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet',
    'name' => 'Rubix\\ML\\NeuralNet\\Network',
    'shortName' => 'Network',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Network
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 24,
    'endLine' => 79,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
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
      'layers' => 
      array (
        'name' => 'layers',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Traversable',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the layers of the network.
 *
 * @return Traversable<Layer>
 */',
        'startLine' => 31,
        'endLine' => 31,
        'startColumn' => 5,
        'endColumn' => 43,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'aliasName' => NULL,
      ),
      'numParams' => 
      array (
        'name' => 'numParams',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the number of trainable parameters in the network.
 *
 * @return int
 */',
        'startLine' => 38,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 38,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'aliasName' => NULL,
      ),
      'initialize' => 
      array (
        'name' => 'initialize',
        'parameters' => 
        array (
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
 * Initialize the parameters of the layers and warm the optimizer cache.
 */',
        'startLine' => 43,
        'endLine' => 43,
        'startColumn' => 5,
        'endColumn' => 40,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'aliasName' => NULL,
      ),
      'infer' => 
      array (
        'name' => 'infer',
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
            'startLine' => 51,
            'endLine' => 51,
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
            'name' => 'NDArray',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Run an inference pass and return the activations at the output layer.
 *
 * @param Dataset $dataset
 * @return NDArray
 */',
        'startLine' => 51,
        'endLine' => 51,
        'startColumn' => 5,
        'endColumn' => 54,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'aliasName' => NULL,
      ),
      'roundtrip' => 
      array (
        'name' => 'roundtrip',
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
                'name' => 'Rubix\\ML\\Datasets\\Labeled',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 57,
            'endLine' => 57,
            'startColumn' => 31,
            'endColumn' => 46,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param Labeled $dataset
 * @return float
 */',
        'startLine' => 57,
        'endLine' => 57,
        'startColumn' => 5,
        'endColumn' => 56,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'aliasName' => NULL,
      ),
      'input' => 
      array (
        'name' => 'input',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\NeuralNet\\Layers\\Input',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the input layer.
 *
 * @return Input
 */',
        'startLine' => 64,
        'endLine' => 64,
        'startColumn' => 5,
        'endColumn' => 36,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'aliasName' => NULL,
      ),
      'hidden' => 
      array (
        'name' => 'hidden',
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
 * Return an array of hidden layers indexed left to right.
 *
 * @return list<Hidden>
 */',
        'startLine' => 71,
        'endLine' => 71,
        'startColumn' => 5,
        'endColumn' => 37,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'aliasName' => NULL,
      ),
      'output' => 
      array (
        'name' => 'output',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\NeuralNet\\Layers\\Output',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the output layer.
 *
 * @return Output
 */',
        'startLine' => 78,
        'endLine' => 78,
        'startColumn' => 5,
        'endColumn' => 38,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Network',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Network',
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