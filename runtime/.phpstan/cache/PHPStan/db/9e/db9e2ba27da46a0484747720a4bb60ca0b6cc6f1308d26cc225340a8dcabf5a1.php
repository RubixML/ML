<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Transformers/SparseRandomProjector.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Transformers\SparseRandomProjector
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-d25fd1e9ac59bc997016e65958498349db6744c129b670eb17d5f865beaed4ad',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Transformers/SparseRandomProjector.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Transformers',
    'name' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
    'shortName' => 'SparseRandomProjector',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Sparse Random Projector
 *
 * A *database-friendly* random projector that samples its random projection matrix from a
 * sparse probabilistic approximation of the Gaussian distribution.
 *
 * References:
 * [1] D. Achlioptas. (2003). Database-friendly random projections: Johnson-Lindenstrauss
 * with binary coins.
 * [2] P. Li at al. (2006). Very Sparse Random Projections.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 29,
    'endLine' => 128,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => 'Rubix\\ML\\Transformers\\GaussianRandomProjector',
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AutotrackRevisions',
    ),
    'immediateConstants' => 
    array (
      'TWO_THIRDS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'name' => 'TWO_THIRDS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2.0 / 3.0',
          'attributes' => 
          array (
            'startLine' => 38,
            'endLine' => 38,
            'startTokenPos' => 73,
            'startFilePos' => 996,
            'endTokenPos' => 77,
            'endFilePos' => 1004,
          ),
        ),
        'docComment' => '/**
 * The decimal representation of the fraction two thirds.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 38,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 43,
      ),
    ),
    'immediateProperties' => 
    array (
      'sparsity' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'name' => 'sparsity',
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
                  'name' => 'float',
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
        'default' => NULL,
        'docComment' => '/**
 * The proportion of zero to non-zero elements in the random projection matrix.
 *
 * @var float|null
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
    ),
    'immediateMethods' => 
    array (
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'dimensions' => 
          array (
            'name' => 'dimensions',
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
            'startLine' => 52,
            'endLine' => 52,
            'startColumn' => 33,
            'endColumn' => 47,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'sparsity' => 
          array (
            'name' => 'sparsity',
            'default' => 
            array (
              'code' => 'self::TWO_THIRDS',
              'attributes' => 
              array (
                'startLine' => 52,
                'endLine' => 52,
                'startTokenPos' => 110,
                'startFilePos' => 1360,
                'endTokenPos' => 112,
                'endFilePos' => 1375,
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
                      'name' => 'float',
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
            'startLine' => 52,
            'endLine' => 52,
            'startColumn' => 50,
            'endColumn' => 84,
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
 * @param int $dimensions
 * @param float|null $sparsity
 * @throws InvalidArgumentException
 */',
        'startLine' => 52,
        'endLine' => 62,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'currentClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'aliasName' => NULL,
      ),
      'fit' => 
      array (
        'name' => 'fit',
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
            'startLine' => 70,
            'endLine' => 70,
            'startColumn' => 25,
            'endColumn' => 40,
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
 * Fit the transformer to a dataset.
 *
 * @param Dataset $dataset
 * @throws InvalidArgumentException
 */',
        'startLine' => 70,
        'endLine' => 115,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'currentClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
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
        'startLine' => 124,
        'endLine' => 127,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
        'currentClassName' => 'Rubix\\ML\\Transformers\\SparseRandomProjector',
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