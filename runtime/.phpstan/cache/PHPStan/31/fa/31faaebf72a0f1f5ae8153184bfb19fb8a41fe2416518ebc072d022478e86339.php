<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Tokenizers/Word.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Tokenizers\Word
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-c06a14f9680a924739d654519f72b741c19862bd7e63d36c6ed78071c8638eb1',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Tokenizers\\Word',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Tokenizers/Word.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Tokenizers',
    'name' => 'Rubix\\ML\\Tokenizers\\Word',
    'shortName' => 'Word',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Word
 *
 * This tokenizer matches words with 1 or more characters.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 14,
    'endLine' => 51,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Tokenizers\\Tokenizer',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'WORD_REGEX' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\Word',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\Word',
        'name' => 'WORD_REGEX',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '"/[\\\\w\'-]+/u"',
          'attributes' => 
          array (
            'startLine' => 21,
            'endLine' => 21,
            'startTokenPos' => 29,
            'startFilePos' => 378,
            'endTokenPos' => 29,
            'endFilePos' => 389,
          ),
        ),
        'docComment' => '/**
 * The regular expression to match words in a sentence.
 *
 * @var string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 21,
        'endLine' => 21,
        'startColumn' => 5,
        'endColumn' => 46,
      ),
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      'tokenize' => 
      array (
        'name' => 'tokenize',
        'parameters' => 
        array (
          'text' => 
          array (
            'name' => 'text',
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
            'startLine' => 31,
            'endLine' => 31,
            'startColumn' => 30,
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
 * Tokenize a blob of text.
 *
 * @internal
 *
 * @param string $text
 * @return list<string>
 */',
        'startLine' => 31,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\Word',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\Word',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\Word',
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
        'startLine' => 47,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\Word',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\Word',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\Word',
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