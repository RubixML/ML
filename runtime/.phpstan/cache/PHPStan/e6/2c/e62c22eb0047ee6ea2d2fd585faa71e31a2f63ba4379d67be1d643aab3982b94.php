<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Graph/Trees/ITree.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Graph\Trees\ITree
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-71352a3fc91ba48fa4d32e02b3bfe1caa0a568a7e11db36a398622b59b5ea795',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Graph/Trees/ITree.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Graph\\Trees',
    'name' => 'Rubix\\ML\\Graph\\Trees\\ITree',
    'shortName' => 'ITree',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * I-Tree
 *
 * The base Isolation Tree implementation with completely random node splitting.
 *
 * References:
 * [1] F. T. Liu et al. (2008). Isolation Forest.
 * [2] F. T. Liu et al. (2011). Isolation-based Anomaly Detection.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 25,
    'endLine' => 192,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Graph\\Trees\\BinaryTree',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'MAX_LEAF_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'name' => 'MAX_LEAF_SIZE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2',
          'attributes' => 
          array (
            'startLine' => 32,
            'endLine' => 32,
            'startTokenPos' => 49,
            'startFilePos' => 697,
            'endTokenPos' => 49,
            'endFilePos' => 697,
          ),
        ),
        'docComment' => '/**
 * The maximum size of a leaf node in the tree.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 5,
        'endColumn' => 38,
      ),
    ),
    'immediateProperties' => 
    array (
      'maxHeight' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'name' => 'maxHeight',
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
 * The maximum depth of a branch before it is forced to terminate.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 39,
        'endLine' => 39,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'root' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'name' => 'root',
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\Isolator',
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
            'startLine' => 46,
            'endLine' => 46,
            'startTokenPos' => 72,
            'startFilePos' => 957,
            'endTokenPos' => 72,
            'endFilePos' => 960,
          ),
        ),
        'docComment' => '/**
 * The root node of the tree.
 *
 * @var Isolator|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 46,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 37,
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
          'maxHeight' => 
          array (
            'name' => 'maxHeight',
            'default' => 
            array (
              'code' => 'PHP_INT_MAX',
              'attributes' => 
              array (
                'startLine' => 52,
                'endLine' => 52,
                'startTokenPos' => 89,
                'startFilePos' => 1098,
                'endTokenPos' => 89,
                'endFilePos' => 1108,
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
            'startLine' => 52,
            'endLine' => 52,
            'startColumn' => 33,
            'endColumn' => 60,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $maxHeight
 * @throws InvalidArgumentException
 */',
        'startLine' => 52,
        'endLine' => 60,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'aliasName' => NULL,
      ),
      'height' => 
      array (
        'name' => 'height',
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
 * Return the height of the tree i.e. the number of levels.
 *
 * @return int
 */',
        'startLine' => 67,
        'endLine' => 70,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'aliasName' => NULL,
      ),
      'balance' => 
      array (
        'name' => 'balance',
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
 * Return the balance factor of the tree. A balanced tree will have
 * a factor of 0 whereas an imbalanced tree will either be positive
 * or negative indicating the direction and degree of the imbalance.
 *
 * @return int
 */',
        'startLine' => 79,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'aliasName' => NULL,
      ),
      'bare' => 
      array (
        'name' => 'bare',
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
 * Is the tree bare?
 *
 * @return bool
 */',
        'startLine' => 89,
        'endLine' => 92,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'aliasName' => NULL,
      ),
      'grow' => 
      array (
        'name' => 'grow',
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
            'startLine' => 100,
            'endLine' => 100,
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
 * Insert a root node and recursively split the dataset until a
 * terminating condition is met.
 *
 * @param Dataset $dataset
 */',
        'startLine' => 100,
        'endLine' => 152,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'aliasName' => NULL,
      ),
      'search' => 
      array (
        'name' => 'search',
        'parameters' => 
        array (
          'sample' => 
          array (
            'name' => 'sample',
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
            'startLine' => 160,
            'endLine' => 160,
            'startColumn' => 28,
            'endColumn' => 40,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\Depth',
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
 * Search the tree for a leaf node.
 *
 * @param list<string|int|float> $sample
 * @return Depth|null
 */',
        'startLine' => 160,
        'endLine' => 191,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\ITree',
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