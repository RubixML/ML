<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Graph/Trees/BallTree.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Graph\Trees\BallTree
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-03b2834d1cd9e841f0702eec093f401cdd31d7edbac766ba226a14c890585c03',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Graph/Trees/BallTree.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Graph\\Trees',
    'name' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
    'shortName' => 'BallTree',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Ball Tree
 *
 * A binary spatial tree that partitions the dataset into successively smaller
 * and tighter *ball* nodes whose boundary are defined by a centroid and radius.
 * Ball Trees work well in higher dimensions since the partitioning schema does
 * not rely on a finite number of 1-dimensional axis aligned splits as with k-d
 * trees.
 *
 * References:
 * [1] S. M. Omohundro. (1989). Five Balltree Construction Algorithms.
 * [2] M. Dolatshah et al. (2015). Ball*-tree: Efficient spatial indexing for
 * constrained nearest-neighbor search in metric spaces.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 36,
    'endLine' => 362,
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
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
 * The maximum number of unique samples that each leaf node can contain.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 43,
        'endLine' => 43,
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
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
        'startLine' => 50,
        'endLine' => 50,
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
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\Ball',
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
            'startLine' => 57,
            'endLine' => 57,
            'startTokenPos' => 110,
            'startFilePos' => 1522,
            'endTokenPos' => 110,
            'endFilePos' => 1525,
          ),
        ),
        'docComment' => '/**
 * The root node of the tree.
 *
 * @var Ball|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 57,
        'endLine' => 57,
        'startColumn' => 5,
        'endColumn' => 33,
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
                'startLine' => 64,
                'endLine' => 64,
                'startTokenPos' => 127,
                'startFilePos' => 1703,
                'endTokenPos' => 127,
                'endFilePos' => 1704,
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
            'startLine' => 64,
            'endLine' => 64,
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
                'startLine' => 64,
                'endLine' => 64,
                'startTokenPos' => 137,
                'startFilePos' => 1727,
                'endTokenPos' => 137,
                'endFilePos' => 1730,
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
            'startLine' => 64,
            'endLine' => 64,
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
        'startLine' => 64,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
        'startLine' => 82,
        'endLine' => 85,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
 * @internal
 *
 * @return int
 */',
        'startLine' => 96,
        'endLine' => 99,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
            'startLine' => 134,
            'endLine' => 134,
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
 * Insert a root node and recursively split the dataset until a terminating
 * condition is met.
 *
 * @internal
 *
 * @param Labeled $dataset
 * @throws InvalidArgumentException
 */',
        'startLine' => 134,
        'endLine' => 169,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
            'startLine' => 181,
            'endLine' => 181,
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
                'startLine' => 181,
                'endLine' => 181,
                'startTokenPos' => 653,
                'startFilePos' => 4778,
                'endTokenPos' => 653,
                'endFilePos' => 4778,
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
            'startLine' => 181,
            'endLine' => 181,
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
 * @param list<string|int|float> $sample
 * @param int $k
 * @throws InvalidArgumentException
 * @return array{list<list<mixed>>,list<mixed>,list<float>}
 */',
        'startLine' => 181,
        'endLine' => 237,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
            'startLine' => 250,
            'endLine' => 250,
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
            'startLine' => 250,
            'endLine' => 250,
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
 * Return all samples, labels, and distances within a given radius of a sample.
 *
 * @internal
 *
 * @param list<string|int|float> $sample
 * @param float $radius
 * @throws InvalidArgumentException
 * @throws \\Rubix\\ML\\Exceptions\\RuntimeException
 * @return array{list<list<mixed>>,list<mixed>,list<float>}
 */',
        'startLine' => 250,
        'endLine' => 288,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
        'startLine' => 295,
        'endLine' => 298,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
            'startLine' => 306,
            'endLine' => 306,
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
 * @param list<string|int|float> $sample
 * @return list<Hypersphere>
 */',
        'startLine' => 306,
        'endLine' => 349,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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
        'startLine' => 358,
        'endLine' => 361,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\BallTree',
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