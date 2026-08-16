<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Tokenizers/WordStemmer.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Tokenizers\WordStemmer
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-a37d2c7922fd8a7dde37a1670e8ec746f41596efc411400f8e25585af667f912',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Tokenizers/WordStemmer.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Tokenizers',
    'name' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
    'shortName' => 'WordStemmer',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Word Stemmer
 *
 * Word Stemmer reduces inflected and derived words to their root form using the Snowball method. For example,
 * the sentence "Majority voting is likely foolish" might stem to "Major vote is like foolish."
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 18,
    'endLine' => 57,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => 'Rubix\\ML\\Tokenizers\\Word',
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
      'stemmer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'name' => 'stemmer',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The word stemmer.
 *
 * @var \\Wamania\\Snowball\\Stemmer\\Stemmer
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 23,
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
          'language' => 
          array (
            'name' => 'language',
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
            'startLine' => 30,
            'endLine' => 30,
            'startColumn' => 33,
            'endColumn' => 48,
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
 * @param string $language
 */',
        'startLine' => 30,
        'endLine' => 33,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'aliasName' => NULL,
      ),
      'tokenize' => 
      array (
        'name' => 'tokenize',
        'parameters' => 
        array (
          'string' => 
          array (
            'name' => 'string',
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
            'startLine' => 41,
            'endLine' => 41,
            'startColumn' => 30,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Tokenize a block of text.
 *
 * @param string $string
 * @return string[]
 */',
        'startLine' => 41,
        'endLine' => 44,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
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
        'startLine' => 53,
        'endLine' => 56,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\WordStemmer',
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