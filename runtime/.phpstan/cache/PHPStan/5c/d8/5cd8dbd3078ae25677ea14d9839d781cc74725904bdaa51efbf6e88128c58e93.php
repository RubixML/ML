<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Graph/Trees/DecisionTree.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Graph\Trees\DecisionTree
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-b96e93261a8cf9283a1b931cfbb03b35341e9c5caeee4e1d6b3f2e1044454d6f',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Graph/Trees/DecisionTree.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Graph\\Trees',
    'name' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
    'shortName' => 'DecisionTree',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 64,
    'docComment' => '/**
 * Decision Tree
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements IteratorAggregate<int,\\Rubix\\ML\\Graph\\Nodes\\Decision>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 35,
    'endLine' => 513,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Graph\\Trees\\BinaryTree',
      1 => 'IteratorAggregate',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'MAX_NODE_LABEL_LENGTH' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'name' => 'MAX_NODE_LABEL_LENGTH',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '30',
          'attributes' => 
          array (
            'startLine' => 42,
            'endLine' => 42,
            'startTokenPos' => 145,
            'startFilePos' => 1011,
            'endTokenPos' => 145,
            'endFilePos' => 1012,
          ),
        ),
        'docComment' => '/**
 * The maximum number of characters before a node label is truncated.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 42,
        'endLine' => 42,
        'startColumn' => 5,
        'endColumn' => 47,
      ),
    ),
    'immediateProperties' => 
    array (
      'maxHeight' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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
        'startLine' => 49,
        'endLine' => 49,
        'startColumn' => 5,
        'endColumn' => 29,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'maxLeafSize' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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
 * The maximum number of samples that a leaf node can contain.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 56,
        'endLine' => 56,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'minPurityIncrease' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'name' => 'minPurityIncrease',
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
 * The minimum increase in purity necessary for a node not to be post pruned.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 63,
        'endLine' => 63,
        'startColumn' => 5,
        'endColumn' => 39,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'root' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\Split',
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
            'startLine' => 70,
            'endLine' => 70,
            'startTokenPos' => 186,
            'startFilePos' => 1569,
            'endTokenPos' => 186,
            'endFilePos' => 1572,
          ),
        ),
        'docComment' => '/**
 * The root node of the tree.
 *
 * @var Split|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 70,
        'endLine' => 70,
        'startColumn' => 5,
        'endColumn' => 34,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'featureCount' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'name' => 'featureCount',
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
                  'name' => 'int',
                  'isIdentifier' => true,
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
            'startLine' => 77,
            'endLine' => 77,
            'startTokenPos' => 200,
            'startFilePos' => 1715,
            'endTokenPos' => 200,
            'endFilePos' => 1718,
          ),
        ),
        'docComment' => '/**
 * The number of feature columns in the training set.
 *
 * @var int<0,max>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 77,
        'endLine' => 77,
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
      'brightness' => 
      array (
        'name' => 'brightness',
        'parameters' => 
        array (
          'color' => 
          array (
            'name' => 'color',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 85,
            'endLine' => 85,
            'startColumn' => 42,
            'endColumn' => 54,
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
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the brightness of a color in hex format.
 *
 * @param string $color
 * @return int
 */',
        'startLine' => 85,
        'endLine' => 94,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 18,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'maxHeight' => 
          array (
            'name' => 'maxHeight',
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
            'startLine' => 105,
            'endLine' => 105,
            'startColumn' => 9,
            'endColumn' => 22,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'maxLeafSize' => 
          array (
            'name' => 'maxLeafSize',
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
            'startLine' => 106,
            'endLine' => 106,
            'startColumn' => 9,
            'endColumn' => 24,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'minPurityIncrease' => 
          array (
            'name' => 'minPurityIncrease',
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
            'startLine' => 107,
            'endLine' => 107,
            'startColumn' => 9,
            'endColumn' => 32,
            'parameterIndex' => 2,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @internal
 *
 * @param int $maxHeight
 * @param int $maxLeafSize
 * @param float $minPurityIncrease
 * @throws \\InvalidArgumentException
 */',
        'startLine' => 104,
        'endLine' => 127,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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
 * Return the number of levels in the tree.
 *
 * @return int
 */',
        'startLine' => 134,
        'endLine' => 137,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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
 * Return a factor that quantifies the skewness of the distribution of nodes in the tree.
 *
 * @return int
 */',
        'startLine' => 144,
        'endLine' => 147,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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
        'startLine' => 156,
        'endLine' => 159,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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
            'startLine' => 169,
            'endLine' => 169,
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
 * Insert a root node and recursively split the dataset a terminating condition is met.
 *
 * @internal
 *
 * @param Labeled $dataset
 * @throws \\InvalidArgumentException
 */',
        'startLine' => 169,
        'endLine' => 237,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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
            'startLine' => 247,
            'endLine' => 247,
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
                  'name' => 'Rubix\\ML\\Graph\\Nodes\\Outcome',
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
 * Search the decision tree for a leaf node and return it.
 *
 * @internal
 *
 * @param list<string|int|float> $sample
 * @return Outcome|null
 */',
        'startLine' => 247,
        'endLine' => 278,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      'featureImportances' => 
      array (
        'name' => 'featureImportances',
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
 * Return the importance scores of each feature column of the training set.
 *
 * @throws \\RuntimeException
 * @return float[]
 */',
        'startLine' => 286,
        'endLine' => 301,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      'getIterator' => 
      array (
        'name' => 'getIterator',
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
 * Return an iterator for all the nodes in the tree starting at the root and traversing depth first.
 *
 * @return \\Generator<BinaryNode>
 */',
        'startLine' => 308,
        'endLine' => 321,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      'exportGraphviz' => 
      array (
        'name' => 'exportGraphviz',
        'parameters' => 
        array (
          'featureNames' => 
          array (
            'name' => 'featureNames',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 334,
                'endLine' => 334,
                'startTokenPos' => 1469,
                'startFilePos' => 8622,
                'endTokenPos' => 1469,
                'endFilePos' => 8625,
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
                      'name' => 'array',
                      'isIdentifier' => true,
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
            'startLine' => 334,
            'endLine' => 334,
            'startColumn' => 36,
            'endColumn' => 62,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'maxDepth' => 
          array (
            'name' => 'maxDepth',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 334,
                'endLine' => 334,
                'startTokenPos' => 1479,
                'startFilePos' => 8645,
                'endTokenPos' => 1479,
                'endFilePos' => 8648,
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
                      'name' => 'int',
                      'isIdentifier' => true,
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
            'startLine' => 334,
            'endLine' => 334,
            'startColumn' => 65,
            'endColumn' => 85,
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
            'name' => 'Rubix\\ML\\Encoding',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Print a representation of the decision tree suitable to render with the
 * graphviz tool. For example, writing it to graph.dot then executing:
 *
 * dot -Tpng graph.dot
 *
 * @param string[]|null $featureNames
 * @param int $maxDepth
 * @throws RuntimeException
 * @return Encoding
 */',
        'startLine' => 334,
        'endLine' => 351,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      'split' => 
      array (
        'name' => 'split',
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
            'startLine' => 359,
            'endLine' => 359,
            'startColumn' => 39,
            'endColumn' => 54,
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
            'name' => 'Rubix\\ML\\Graph\\Nodes\\Split',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Find a split point for a given subset of the training set.
 *
 * @param Labeled $dataset
 * @return Split
 */',
        'startLine' => 359,
        'endLine' => 359,
        'startColumn' => 5,
        'endColumn' => 64,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 66,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      'terminate' => 
      array (
        'name' => 'terminate',
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
            'startLine' => 367,
            'endLine' => 367,
            'startColumn' => 43,
            'endColumn' => 58,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Terminate a branch with an outcome node.
 *
 * @param Labeled $dataset
 * @return Outcome
 */',
        'startLine' => 367,
        'endLine' => 367,
        'startColumn' => 5,
        'endColumn' => 60,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 66,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      'impurity' => 
      array (
        'name' => 'impurity',
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
            'startLine' => 375,
            'endLine' => 375,
            'startColumn' => 42,
            'endColumn' => 54,
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
 * Calculate the impurity of a set of labels.
 *
 * @param list<string|int> $labels
 * @return float
 */',
        'startLine' => 375,
        'endLine' => 375,
        'startColumn' => 5,
        'endColumn' => 64,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 66,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      'splitImpurity' => 
      array (
        'name' => 'splitImpurity',
        'parameters' => 
        array (
          'subsets' => 
          array (
            'name' => 'subsets',
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
            'startLine' => 383,
            'endLine' => 383,
            'startColumn' => 38,
            'endColumn' => 51,
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
 * Calculate the impurity of a given split.
 *
 * @param array{Labeled,Labeled} $subsets
 * @return float
 */',
        'startLine' => 383,
        'endLine' => 400,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'aliasName' => NULL,
      ),
      '_exportGraphviz' => 
      array (
        'name' => '_exportGraphviz',
        'parameters' => 
        array (
          'carry' => 
          array (
            'name' => 'carry',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => true,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 415,
            'endLine' => 415,
            'startColumn' => 9,
            'endColumn' => 22,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'nodesCounter' => 
          array (
            'name' => 'nodesCounter',
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
            'byRef' => true,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 416,
            'endLine' => 416,
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 1,
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
                'name' => 'Rubix\\ML\\Graph\\Nodes\\BinaryNode',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 417,
            'endLine' => 417,
            'startColumn' => 9,
            'endColumn' => 24,
            'parameterIndex' => 2,
            'isOptional' => false,
          ),
          'maxDepth' => 
          array (
            'name' => 'maxDepth',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 418,
                'endLine' => 418,
                'startTokenPos' => 1798,
                'startFilePos' => 10841,
                'endTokenPos' => 1798,
                'endFilePos' => 10844,
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
                      'name' => 'int',
                      'isIdentifier' => true,
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
            'startLine' => 418,
            'endLine' => 418,
            'startColumn' => 9,
            'endColumn' => 29,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
          'featureNames' => 
          array (
            'name' => 'featureNames',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 419,
                'endLine' => 419,
                'startTokenPos' => 1808,
                'startFilePos' => 10878,
                'endTokenPos' => 1808,
                'endFilePos' => 10881,
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
                      'name' => 'array',
                      'isIdentifier' => true,
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
            'startLine' => 419,
            'endLine' => 419,
            'startColumn' => 9,
            'endColumn' => 35,
            'parameterIndex' => 4,
            'isOptional' => true,
          ),
          'parentId' => 
          array (
            'name' => 'parentId',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 420,
                'endLine' => 420,
                'startTokenPos' => 1818,
                'startFilePos' => 10909,
                'endTokenPos' => 1818,
                'endFilePos' => 10912,
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
                      'name' => 'int',
                      'isIdentifier' => true,
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
            'startLine' => 420,
            'endLine' => 420,
            'startColumn' => 9,
            'endColumn' => 29,
            'parameterIndex' => 5,
            'isOptional' => true,
          ),
          'leftRight' => 
          array (
            'name' => 'leftRight',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 421,
                'endLine' => 421,
                'startTokenPos' => 1828,
                'startFilePos' => 10941,
                'endTokenPos' => 1828,
                'endFilePos' => 10944,
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
                      'name' => 'int',
                      'isIdentifier' => true,
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
            'startLine' => 421,
            'endLine' => 421,
            'startColumn' => 9,
            'endColumn' => 30,
            'parameterIndex' => 6,
            'isOptional' => true,
          ),
          'depth' => 
          array (
            'name' => 'depth',
            'default' => 
            array (
              'code' => '0',
              'attributes' => 
              array (
                'startLine' => 422,
                'endLine' => 422,
                'startTokenPos' => 1837,
                'startFilePos' => 10968,
                'endTokenPos' => 1837,
                'endFilePos' => 10968,
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
            'startLine' => 422,
            'endLine' => 422,
            'startColumn' => 9,
            'endColumn' => 22,
            'parameterIndex' => 7,
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
 * Recursive function to print out the decision rule at each node using preorder traversal.
 *
 * @param string $carry
 * @param int $nodesCounter
 * @param BinaryNode $node
 * @param int $maxDepth
 * @param string[]|null $featureNames
 * @param int|null $parentId
 * @param int|null $leftRight
 * @param int $depth
 */',
        'startLine' => 414,
        'endLine' => 512,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Graph\\Trees',
        'declaringClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'implementingClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
        'currentClassName' => 'Rubix\\ML\\Graph\\Trees\\DecisionTree',
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