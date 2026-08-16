<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/FeedForward.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\FeedForward
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-2efc935f4c859c61244902a4556246e12ee259f536320d9367a287a7190e1603',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/FeedForward.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet',
    'name' => 'Rubix\\ML\\NeuralNet\\FeedForward',
    'shortName' => 'FeedForward',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Feed Forward
 *
 * A feed forward neural network implementation consisting of an input and
 * output layer and any number of intermediate hidden layers.
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
    'startLine' => 37,
    'endLine' => 289,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\NeuralNet\\Network',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'input' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'name' => 'input',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\NeuralNet\\Layers\\Input',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The input layer to the network.
 *
 * @var Input
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
      'hidden' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'name' => 'hidden',
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
            'startLine' => 51,
            'endLine' => 53,
            'startTokenPos' => 132,
            'startFilePos' => 1250,
            'endTokenPos' => 136,
            'endFilePos' => 1267,
          ),
        ),
        'docComment' => '/**
 * The hidden layers of the network.
 *
 * @var list<Hidden>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 51,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 6,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'backPass' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'name' => 'backPass',
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
            'startTokenPos' => 149,
            'startFilePos' => 1418,
            'endTokenPos' => 153,
            'endFilePos' => 1435,
          ),
        ),
        'docComment' => '/**
 * The pathing of the backward pass through the hidden layers.
 *
 * @var list<Hidden>
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
      'output' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'name' => 'output',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\NeuralNet\\Layers\\Output',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The output layer of the network.
 *
 * @var Output
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 69,
        'endLine' => 69,
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
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
 * The gradient descent optimizer used to train the network.
 *
 * @var Optimizer
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 76,
        'endLine' => 76,
        'startColumn' => 5,
        'endColumn' => 35,
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
          'input' => 
          array (
            'name' => 'input',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\NeuralNet\\Layers\\Input',
                'isIdentifier' => false,
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
            'startColumn' => 33,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'hidden' => 
          array (
            'name' => 'hidden',
            'default' => NULL,
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
            'startLine' => 84,
            'endLine' => 84,
            'startColumn' => 47,
            'endColumn' => 59,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'output' => 
          array (
            'name' => 'output',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\NeuralNet\\Layers\\Output',
                'isIdentifier' => false,
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
            'startColumn' => 62,
            'endColumn' => 75,
            'parameterIndex' => 2,
            'isOptional' => false,
          ),
          'optimizer' => 
          array (
            'name' => 'optimizer',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Optimizer',
                'isIdentifier' => false,
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
            'startColumn' => 78,
            'endColumn' => 97,
            'parameterIndex' => 3,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param Input $input
 * @param Hidden[] $hidden
 * @param Output $output
 * @param Optimizer $optimizer
 */',
        'startLine' => 84,
        'endLine' => 100,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
        'startLine' => 107,
        'endLine' => 110,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
        'startLine' => 117,
        'endLine' => 120,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
        'startLine' => 127,
        'endLine' => 130,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'aliasName' => NULL,
      ),
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
 * Return all the layers in the network.
 *
 * @return Traversable<Layer>
 */',
        'startLine' => 137,
        'endLine' => 144,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
        'startLine' => 151,
        'endLine' => 164,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
        'startLine' => 169,
        'endLine' => 186,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
            'startLine' => 194,
            'endLine' => 194,
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
        'startLine' => 194,
        'endLine' => 209,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
            'startLine' => 218,
            'endLine' => 218,
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
 * Perform a forward and backward pass of the network in one call. Returns
 * the loss from the backward pass.
 *
 * @param Labeled $dataset
 * @return float
 */',
        'startLine' => 218,
        'endLine' => 227,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'aliasName' => NULL,
      ),
      'feed' => 
      array (
        'name' => 'feed',
        'parameters' => 
        array (
          'input' => 
          array (
            'name' => 'input',
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
            'startLine' => 235,
            'endLine' => 235,
            'startColumn' => 26,
            'endColumn' => 39,
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
 * Feed a batch through the network and return a matrix of activations at the output later.
 *
 * @param NDArray $input
 * @return NDArray
 */',
        'startLine' => 235,
        'endLine' => 242,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'aliasName' => NULL,
      ),
      'backpropagate' => 
      array (
        'name' => 'backpropagate',
        'parameters' => 
        array (
          'labels' => 
          array (
            'name' => 'labels',
            'default' => NULL,
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
            'startLine' => 250,
            'endLine' => 250,
            'startColumn' => 35,
            'endColumn' => 47,
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
 * Backpropagate the gradient of the cost function and return the loss.
 *
 * @param list<string|int|float> $labels
 * @return float
 */',
        'startLine' => 250,
        'endLine' => 259,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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
 * @return Encoding
 */',
        'startLine' => 266,
        'endLine' => 288,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\FeedForward',
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