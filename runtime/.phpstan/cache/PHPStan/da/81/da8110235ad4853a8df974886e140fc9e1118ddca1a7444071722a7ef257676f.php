<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Transformers/TfIdfTransformer.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Transformers\TfIdfTransformer
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-e9fdfddb5c02c01f5dff2e33c5d0d5361de76a09f190a1f0120f1e28a0b6e5e5',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Transformers/TfIdfTransformer.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Transformers',
    'name' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
    'shortName' => 'TfIdfTransformer',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * TF-IDF Transformer
 *
 * Term Frequency - Inverse Document Frequency is a measure of how important a word is to
 * a document. The TF-IDF value increases with the number of times a word appears in a document
 * and is offset by the frequency of the word in the corpus.
 *
 * > **Note**: TF-IDF Transformer assumes that its input is made up of term frequency
 * vectors such as those created by Word Count or Token Hashing Vectorizer.
 *
 * References:
 * [1] S. Robertson. (2003). Understanding Inverse Document Frequency: On theoretical
 * arguments for IDF.
 * [2] C. D. Manning et al. (2009). An Introduction to Information Retrieval.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 38,
    'endLine' => 242,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Transformers\\Transformer',
      1 => 'Rubix\\ML\\Transformers\\Stateful',
      2 => 'Rubix\\ML\\Transformers\\Elastic',
      3 => 'Rubix\\ML\\Transformers\\Reversible',
      4 => 'Rubix\\ML\\Persistable',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AutotrackRevisions',
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'smoothing' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'name' => 'smoothing',
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
 * The amount of additive (Laplace) smoothing to add to the IDFs.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 47,
        'endLine' => 47,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'sublinear' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'name' => 'sublinear',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * Should we apply a sub-linear function to dampen the effect of recurring tokens?
 *
 * @var bool
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 54,
        'endLine' => 54,
        'startColumn' => 5,
        'endColumn' => 30,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'dfs' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'name' => 'dfs',
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
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 61,
            'endLine' => 61,
            'startTokenPos' => 129,
            'startFilePos' => 1827,
            'endTokenPos' => 129,
            'endFilePos' => 1830,
          ),
        ),
        'docComment' => '/**
 * The document frequencies of each word i.e. the number of times a word appeared in a document.
 *
 * @var int[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 61,
        'endLine' => 61,
        'startColumn' => 5,
        'endColumn' => 33,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'idfs' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'name' => 'idfs',
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
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 68,
            'endLine' => 68,
            'startTokenPos' => 143,
            'startFilePos' => 1976,
            'endTokenPos' => 143,
            'endFilePos' => 1979,
          ),
        ),
        'docComment' => '/**
 * The inverse document frequencies for each feature column.
 *
 * @var float[]|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 68,
        'endLine' => 68,
        'startColumn' => 5,
        'endColumn' => 34,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'n' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'name' => 'n',
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
        'default' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 75,
            'endLine' => 75,
            'startTokenPos' => 156,
            'startFilePos' => 2116,
            'endTokenPos' => 156,
            'endFilePos' => 2116,
          ),
        ),
        'docComment' => '/**
 * The number of documents (samples) that have been fitted so far.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 75,
        'endLine' => 75,
        'startColumn' => 5,
        'endColumn' => 25,
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
          'smoothing' => 
          array (
            'name' => 'smoothing',
            'default' => 
            array (
              'code' => '1.0',
              'attributes' => 
              array (
                'startLine' => 82,
                'endLine' => 82,
                'startTokenPos' => 173,
                'startFilePos' => 2288,
                'endTokenPos' => 173,
                'endFilePos' => 2290,
              ),
            ),
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
            'startLine' => 82,
            'endLine' => 82,
            'startColumn' => 33,
            'endColumn' => 54,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'sublinear' => 
          array (
            'name' => 'sublinear',
            'default' => 
            array (
              'code' => 'false',
              'attributes' => 
              array (
                'startLine' => 82,
                'endLine' => 82,
                'startTokenPos' => 182,
                'startFilePos' => 2311,
                'endTokenPos' => 182,
                'endFilePos' => 2315,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'bool',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 82,
            'endLine' => 82,
            'startColumn' => 57,
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
 * @param float $smoothing
 * @param bool $sublinear
 * @throws InvalidArgumentException
 */',
        'startLine' => 82,
        'endLine' => 91,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'aliasName' => NULL,
      ),
      'compatibility' => 
      array (
        'name' => 'compatibility',
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
 * Return the data types that this transformer is compatible with.
 *
 * @internal
 *
 * @return list<DataType>
 */',
        'startLine' => 100,
        'endLine' => 105,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'aliasName' => NULL,
      ),
      'fitted' => 
      array (
        'name' => 'fitted',
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
 * Is the transformer fitted?
 *
 * @return bool
 */',
        'startLine' => 112,
        'endLine' => 115,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'aliasName' => NULL,
      ),
      'dfs' => 
      array (
        'name' => 'dfs',
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
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the document frequencies calculated during fitting.
 *
 * @return int[]|null
 */',
        'startLine' => 122,
        'endLine' => 125,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
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
            'startLine' => 132,
            'endLine' => 132,
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
 */',
        'startLine' => 132,
        'endLine' => 138,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'aliasName' => NULL,
      ),
      'update' => 
      array (
        'name' => 'update',
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
            'startLine' => 146,
            'endLine' => 146,
            'startColumn' => 28,
            'endColumn' => 43,
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
 * Update the fitting of the transformer.
 *
 * @param Dataset $dataset
 * @throws InvalidArgumentException
 */',
        'startLine' => 146,
        'endLine' => 178,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'aliasName' => NULL,
      ),
      'transform' => 
      array (
        'name' => 'transform',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
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
            'byRef' => true,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 186,
            'endLine' => 186,
            'startColumn' => 31,
            'endColumn' => 45,
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
 * Transform the dataset in place.
 *
 * @param list<list<mixed>> $samples
 * @throws RuntimeException
 */',
        'startLine' => 186,
        'endLine' => 203,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'aliasName' => NULL,
      ),
      'reverseTransform' => 
      array (
        'name' => 'reverseTransform',
        'parameters' => 
        array (
          'samples' => 
          array (
            'name' => 'samples',
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
            'byRef' => true,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 211,
            'endLine' => 211,
            'startColumn' => 38,
            'endColumn' => 52,
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
 * Perform the reverse transformation to the samples.
 *
 * @param list<list<mixed>> $samples
 * @throws RuntimeException
 */',
        'startLine' => 211,
        'endLine' => 228,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
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
        'startLine' => 237,
        'endLine' => 241,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
        'currentClassName' => 'Rubix\\ML\\Transformers\\TfIdfTransformer',
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