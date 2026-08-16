<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Graph/Nodes/Traits/HasBinaryChildrenTrait.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Graph\Nodes\Traits\HasBinaryChildrenTrait
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-2af4c91afd0fdb5b4376284e79949b6ff7c3d65185a1b86031b80c52ca83077c',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Graph/Nodes/Traits/HasBinaryChildrenTrait.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Graph\\Nodes\\Traits',
    'name' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
    'shortName' => 'HasBinaryChildrenTrait',
    'isInterface' => false,
    'isTrait' => true,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Has Binary Children Trait
 *
 * A node of a binary tree i.e a tree whose nodes have a maximum of two immediate children.
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
    'startLine' => 19,
    'endLine' => 114,
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
      'left' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'name' => 'left',
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\BinaryNode',
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
            'startLine' => 26,
            'endLine' => 26,
            'startTokenPos' => 36,
            'startFilePos' => 486,
            'endTokenPos' => 36,
            'endFilePos' => 489,
          ),
        ),
        'docComment' => '/**
 * The left child node.
 *
 * @var BinaryNode|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 26,
        'endLine' => 26,
        'startColumn' => 5,
        'endColumn' => 39,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'right' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'name' => 'right',
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\BinaryNode',
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
            'startLine' => 33,
            'endLine' => 33,
            'startTokenPos' => 50,
            'startFilePos' => 608,
            'endTokenPos' => 50,
            'endFilePos' => 611,
          ),
        ),
        'docComment' => '/**
 * The right child node.
 *
 * @var BinaryNode|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 33,
        'endLine' => 33,
        'startColumn' => 5,
        'endColumn' => 40,
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
      'children' => 
      array (
        'name' => 'children',
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
 * Return the children of this node in a generator.
 *
 * @return \\Generator<BinaryNode>
 */',
        'startLine' => 40,
        'endLine' => 49,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'aliasName' => NULL,
      ),
      'left' => 
      array (
        'name' => 'left',
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\BinaryNode',
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
 * Return the left child node.
 *
 * @return BinaryNode|null
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
        'namespace' => 'Rubix\\ML\\Graph\\Nodes\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'aliasName' => NULL,
      ),
      'right' => 
      array (
        'name' => 'right',
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\BinaryNode',
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
 * Return the right child node.
 *
 * @return BinaryNode|null
 */',
        'startLine' => 66,
        'endLine' => 69,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
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
 * Recursive function to determine the height of the node in the tree.
 *
 * @return int
 */',
        'startLine' => 76,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
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
 * The balance factor of the node. Negative numbers indicate a lean to the left, positive
 * to the right, and 0 is perfectly balanced.
 *
 * @return int
 */',
        'startLine' => 90,
        'endLine' => 93,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'aliasName' => NULL,
      ),
      'attachLeft' => 
      array (
        'name' => 'attachLeft',
        'parameters' => 
        array (
          'node' => 
          array (
            'name' => 'node',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 100,
                'endLine' => 100,
                'startTokenPos' => 306,
                'startFilePos' => 2062,
                'endTokenPos' => 306,
                'endFilePos' => 2065,
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
                      'name' => 'Rubix\\ML\\Graph\\Nodes\\BinaryNode',
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
            'startLine' => 100,
            'endLine' => 100,
            'startColumn' => 32,
            'endColumn' => 55,
            'parameterIndex' => 0,
            'isOptional' => true,
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
 * Set the left child node.
 *
 * @param BinaryNode|null $node
 */',
        'startLine' => 100,
        'endLine' => 103,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'aliasName' => NULL,
      ),
      'attachRight' => 
      array (
        'name' => 'attachRight',
        'parameters' => 
        array (
          'node' => 
          array (
            'name' => 'node',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 110,
                'endLine' => 110,
                'startTokenPos' => 341,
                'startFilePos' => 2261,
                'endTokenPos' => 341,
                'endFilePos' => 2264,
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
                      'name' => 'Rubix\\ML\\Graph\\Nodes\\BinaryNode',
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
            'startLine' => 110,
            'endLine' => 110,
            'startColumn' => 33,
            'endColumn' => 56,
            'parameterIndex' => 0,
            'isOptional' => true,
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
 * Set the right child node.
 *
 * @param BinaryNode|null $node
 */',
        'startLine' => 110,
        'endLine' => 113,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Nodes\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
        'currentClassName' => 'Rubix\\ML\\Graph\\Nodes\\Traits\\HasBinaryChildrenTrait',
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