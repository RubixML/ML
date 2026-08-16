<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Graph/Trees/KDTree.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Graph\Trees\KDTree
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-da698052017b294700bc8c5d888ad16eef1e3634fac0ccbcb11f54bd2bf4a8ff',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Graph/Trees/KDTree.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Graph\\Trees',
    'name' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
    'shortName' => 'KDTree',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * K-d Tree
 *
 * A multi-dimensional binary search tree for fast nearest neighbor queries.
 * The K-d tree construction algorithm separates data points into bounded
 * hypercubes or *bounding boxes* that are used to prune off nodes during
 * nearest neighbor and range searches.
 *
 * [1] J. L. Bentley. (1975). Multidimensional Binary Search Trees Used for
 * Associative Searching.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 38,
    'endLine' => 373,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Graph\\Trees\\BinaryTree',
      1 => 'Rubix\\ML\\Graph\\Trees\\Spatial',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'maxLeafSize' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'name' => 'maxLeafSize',
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
 * The maximum number of samples that each neighborhood node can contain.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 45,
        'endLine' => 45,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'kernel' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'name' => 'kernel',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Kernels\\Distance\\Distance',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The distance function to use when computing the distances.
 *
 * @var Distance
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 52,
        'endLine' => 52,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'root' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\Box',
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
            'startLine' => 59,
            'endLine' => 59,
            'startTokenPos' => 143,
            'startFilePos' => 1463,
            'endTokenPos' => 143,
            'endFilePos' => 1466,
          ),
        ),
        'docComment' => '/**
 * The root node of the tree.
 *
 * @var Box|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 59,
        'endLine' => 59,
        'startColumn' => 5,
        'endColumn' => 32,
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
          'maxLeafSize' => 
          array (
            'name' => 'maxLeafSize',
            'default' => 
            array (
              'code' => '30',
              'attributes' => 
              array (
                'startLine' => 66,
                'endLine' => 66,
                'startTokenPos' => 160,
                'startFilePos' => 1644,
                'endTokenPos' => 160,
                'endFilePos' => 1645,
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
            'startLine' => 66,
            'endLine' => 66,
            'startColumn' => 33,
            'endColumn' => 53,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'kernel' => 
          array (
            'name' => 'kernel',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 66,
                'endLine' => 66,
                'startTokenPos' => 170,
                'startFilePos' => 1668,
                'endTokenPos' => 170,
                'endFilePos' => 1671,
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
                      'name' => 'Rubix\\ML\\Kernels\\Distance\\Distance',
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
            'startLine' => 66,
            'endLine' => 66,
            'startColumn' => 56,
            'endColumn' => 79,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $maxLeafSize
 * @param Distance|null $kernel
 * @throws InvalidArgumentException
 */',
        'startLine' => 66,
        'endLine' => 85,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
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
 * @internal
 *
 * @return int
 */',
        'startLine' => 94,
        'endLine' => 97,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
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
 * Return the balance factor of the tree. A balanced tree will have  a factor of 0 whereas
 * an imbalanced tree will either be positive or negative indicating the direction and
 * degree of the imbalance.
 *
 * @internal
 *
 * @return int
 */',
        'startLine' => 108,
        'endLine' => 111,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
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
 * @internal
 *
 * @return bool
 */',
        'startLine' => 120,
        'endLine' => 123,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'aliasName' => NULL,
      ),
      'kernel' => 
      array (
        'name' => 'kernel',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Kernels\\Distance\\Distance',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the distance kernel used to compute distances.
 *
 * @internal
 *
 * @return Distance
 */',
        'startLine' => 132,
        'endLine' => 135,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
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
            'startLine' => 145,
            'endLine' => 145,
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
 * Insert a root node and recursively split the dataset until a terminating condition is met.
 *
 * @internal
 *
 * @param Labeled $dataset
 * @throws InvalidArgumentException
 */',
        'startLine' => 145,
        'endLine' => 184,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'aliasName' => NULL,
      ),
      'nearest' => 
      array (
        'name' => 'nearest',
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
            'startLine' => 196,
            'endLine' => 196,
            'startColumn' => 29,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'k' => 
          array (
            'name' => 'k',
            'default' => 
            array (
              'code' => '1',
              'attributes' => 
              array (
                'startLine' => 196,
                'endLine' => 196,
                'startTokenPos' => 774,
                'startFilePos' => 5264,
                'endTokenPos' => 774,
                'endFilePos' => 5264,
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
            'startLine' => 196,
            'endLine' => 196,
            'startColumn' => 44,
            'endColumn' => 53,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
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
 * Run a k nearest neighbors search and return the samples, labels, and distances in a 3-tuple.
 *
 * @internal
 *
 * @param list<int|float> $sample
 * @param int $k
 * @throws InvalidArgumentException
 * @return array{list<list<mixed>>,list<mixed>,list<float>}
 */',
        'startLine' => 196,
        'endLine' => 252,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'aliasName' => NULL,
      ),
      'range' => 
      array (
        'name' => 'range',
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
            'startLine' => 264,
            'endLine' => 264,
            'startColumn' => 27,
            'endColumn' => 39,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'radius' => 
          array (
            'name' => 'radius',
            'default' => NULL,
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
            'startLine' => 264,
            'endLine' => 264,
            'startColumn' => 42,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Run a range search over every cluster within radius and return the samples, labels and distances in a 3-tuple.
 *
 * @internal
 *
 * @param list<int|float> $sample
 * @param float $radius
 * @throws InvalidArgumentException
 * @return array{list<list<mixed>>,list<mixed>,list<float>}
 */',
        'startLine' => 264,
        'endLine' => 302,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'aliasName' => NULL,
      ),
      'destroy' => 
      array (
        'name' => 'destroy',
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
 * Destroy the tree.
 *
 * @internal
 */',
        'startLine' => 309,
        'endLine' => 312,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'aliasName' => NULL,
      ),
      'path' => 
      array (
        'name' => 'path',
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
            'startLine' => 320,
            'endLine' => 320,
            'startColumn' => 29,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the path of a sample taken from the root node to a leaf node in an array.
 *
 * @param list<int|float> $sample
 * @return list<\\Rubix\\ML\\Graph\\Nodes\\Node|null>
 */',
        'startLine' => 320,
        'endLine' => 339,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'aliasName' => NULL,
      ),
      'minDistance' => 
      array (
        'name' => 'minDistance',
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
            'startLine' => 349,
            'endLine' => 349,
            'startColumn' => 34,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'node' => 
          array (
            'name' => 'node',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Graph\\Nodes\\Hypercube',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 349,
            'endLine' => 349,
            'startColumn' => 49,
            'endColumn' => 63,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Compute the lower bound of the distance between a sample and any point contained within the
 * bounding box of a hypercube.
 *
 * @param list<int|float> $sample
 * @param Hypercube $node
 * @return float
 */',
        'startLine' => 349,
        'endLine' => 360,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 4,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
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
        'startLine' => 369,
        'endLine' => 372,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\KDTree',
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