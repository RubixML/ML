<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Tokenizers/Sentence.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Tokenizers\Sentence
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-9aa35e113fef74c3db6dbeda95e18a826fba5144b6a775ff0e1472bc3348c4a8',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Tokenizers\\Sentence',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Tokenizers/Sentence.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Tokenizers',
    'name' => 'Rubix\\ML\\Tokenizers\\Sentence',
    'shortName' => 'Sentence',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Sentence
 *
 * This tokenizer matches sentences starting with a letter and ending with a punctuation mark.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 16,
    'endLine' => 47,
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
      'SENTENCE_REGEX' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\Sentence',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\Sentence',
        'name' => 'SENTENCE_REGEX',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/(?<=[\\.\\?\\⸮\\؟\\!\\n])(\\n+|\\s+|\\b(?=\\D))(?=[\\\'\\"\\w])/iu\'',
          'attributes' => 
          array (
            'startLine' => 23,
            'endLine' => 23,
            'startTokenPos' => 36,
            'startFilePos' => 460,
            'endTokenPos' => 36,
            'endFilePos' => 517,
          ),
        ),
        'docComment' => '/**
 * The regular expression to match sentences in a blob of text.
 *
 * @var string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 23,
        'endLine' => 23,
        'startColumn' => 5,
        'endColumn' => 96,
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
 * @param string $text
 * @return list<string>
 */',
        'startLine' => 31,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\Sentence',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\Sentence',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\Sentence',
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
        'startLine' => 43,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Tokenizers',
        'declaringClassName' => 'Rubix\\ML\\Tokenizers\\Sentence',
        'implementingClassName' => 'Rubix\\ML\\Tokenizers\\Sentence',
        'currentClassName' => 'Rubix\\ML\\Tokenizers\\Sentence',
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