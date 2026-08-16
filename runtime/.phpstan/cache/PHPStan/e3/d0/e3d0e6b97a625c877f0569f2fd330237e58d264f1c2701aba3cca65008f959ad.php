<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Transformers/RegexFilter.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Transformers\RegexFilter
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-529e145c1811e973dfd4b77e2cfc71b0c008fa3d2b26d238334c0ddb8791e37b',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Transformers/RegexFilter.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Transformers',
    'name' => 'Rubix\\ML\\Transformers\\RegexFilter',
    'shortName' => 'RegexFilter',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Regex Filter
 *
 * Filters the text features of a dataset by matching and removing patterns from a list of regular expressions.
 *
 * References:
 * [1] J. Gruber. (2009). A Liberal, Accurate Regex Pattern for Matching URLs.
 * [2] J. Gruber. (2010). An Improved Liberal, Accurate Regex Pattern for Matching URLs.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 27,
    'endLine' => 173,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Transformers\\Transformer',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'EMAIL' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'EMAIL',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/[a-z0-9_\\-\\+\\.]+@[a-z0-9\\-]+\\.([a-z]{2,4})(?:\\.[a-z]{2})?/i\'',
          'attributes' => 
          array (
            'startLine' => 34,
            'endLine' => 34,
            'startTokenPos' => 74,
            'startFilePos' => 819,
            'endTokenPos' => 74,
            'endFilePos' => 880,
          ),
        ),
        'docComment' => '/**
 * A pattern to match email addresses.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 34,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 88,
      ),
      'URL' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'URL',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => 'self::GRUBER_1',
          'attributes' => 
          array (
            'startLine' => 41,
            'endLine' => 41,
            'startTokenPos' => 87,
            'startFilePos' => 998,
            'endTokenPos' => 89,
            'endFilePos' => 1011,
          ),
        ),
        'docComment' => '/**
 * The default URL matching pattern.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 41,
        'endLine' => 41,
        'startColumn' => 5,
        'endColumn' => 38,
      ),
      'GRUBER_1' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'GRUBER_1',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'%\\b(([\\w-]+://?|www[.])[^\\s()<>]+(?:\\([\\w\\d]+\\)|([^[:punct:]\\s]|/)))%s\'',
          'attributes' => 
          array (
            'startLine' => 48,
            'endLine' => 48,
            'startTokenPos' => 102,
            'startFilePos' => 1142,
            'endTokenPos' => 102,
            'endFilePos' => 1213,
          ),
        ),
        'docComment' => '/**
 * The original Gruber URL matching pattern.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 48,
        'endLine' => 48,
        'startColumn' => 5,
        'endColumn' => 101,
      ),
      'GRUBER_2' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'GRUBER_2',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'%(?xi)\\b((?:https?://|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\))+(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]{};:\\\'".,<>?«»“”‘’]))%s\'',
          'attributes' => 
          array (
            'startLine' => 55,
            'endLine' => 55,
            'startTokenPos' => 115,
            'startFilePos' => 1344,
            'endTokenPos' => 115,
            'endFilePos' => 1541,
          ),
        ),
        'docComment' => '/**
 * The improved Gruber URL matching pattern.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 55,
        'endLine' => 55,
        'startColumn' => 5,
        'endColumn' => 227,
      ),
      'EXTRA_CHARACTERS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'EXTRA_CHARACTERS',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/([^\\w\\s])(?=[^\\w\\s]*\\1)/u\'',
          'attributes' => 
          array (
            'startLine' => 62,
            'endLine' => 62,
            'startTokenPos' => 128,
            'startFilePos' => 1743,
            'endTokenPos' => 128,
            'endFilePos' => 1770,
          ),
        ),
        'docComment' => '/**
 * Matches consecutively repeated non word or number characters such as punctuation and special characters.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 62,
        'endLine' => 62,
        'startColumn' => 5,
        'endColumn' => 65,
      ),
      'EXTRA_WORDS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'EXTRA_WORDS',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/\\b(\\w+)(?=\\s+\\1+\\b)/ui\'',
          'attributes' => 
          array (
            'startLine' => 69,
            'endLine' => 69,
            'startTokenPos' => 141,
            'startFilePos' => 1900,
            'endTokenPos' => 141,
            'endFilePos' => 1924,
          ),
        ),
        'docComment' => '/**
 * Matches consecutively repeated words.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 69,
        'endLine' => 69,
        'startColumn' => 5,
        'endColumn' => 57,
      ),
      'EXTRA_WHITESPACE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'EXTRA_WHITESPACE',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/\\s(?=\\s+)/u\'',
          'attributes' => 
          array (
            'startLine' => 76,
            'endLine' => 76,
            'startTokenPos' => 154,
            'startFilePos' => 2075,
            'endTokenPos' => 154,
            'endFilePos' => 2088,
          ),
        ),
        'docComment' => '/**
 * Matches consecutively repeated whitespace characters.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 76,
        'endLine' => 76,
        'startColumn' => 5,
        'endColumn' => 51,
      ),
      'EMOJIS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'EMOJIS',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/[\\x{1F300}-\\x{1F5FF}\\x{1F900}-\\x{1F9FF}\\x{1F600}-\\x{1F64F}\\x{1F680}-\\x{1F6FF}\\x{2600}-\\x{26FF}\\x{2700}-\\x{27BF}]/u\'',
          'attributes' => 
          array (
            'startLine' => 83,
            'endLine' => 83,
            'startTokenPos' => 167,
            'startFilePos' => 2210,
            'endTokenPos' => 167,
            'endFilePos' => 2326,
          ),
        ),
        'docComment' => '/**
 * A pattern to match unicode emojis.
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 83,
        'endLine' => 83,
        'startColumn' => 5,
        'endColumn' => 144,
      ),
      'MENTION' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'MENTION',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/(@\\w+)/u\'',
          'attributes' => 
          array (
            'startLine' => 90,
            'endLine' => 90,
            'startTokenPos' => 180,
            'startFilePos' => 2472,
            'endTokenPos' => 180,
            'endFilePos' => 2482,
          ),
        ),
        'docComment' => '/**
 * A pattern to match Twitter-style mentions (ex. @RubixML).
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 90,
        'endLine' => 90,
        'startColumn' => 5,
        'endColumn' => 39,
      ),
      'HASHTAG' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'HASHTAG',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'/(#\\w+)/u\'',
          'attributes' => 
          array (
            'startLine' => 97,
            'endLine' => 97,
            'startTokenPos' => 193,
            'startFilePos' => 2636,
            'endTokenPos' => 193,
            'endFilePos' => 2646,
          ),
        ),
        'docComment' => '/**
 * A pattern to match Twitter-style hashtags (ex. #MachineLearning).
 *
 * @var literal-string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 97,
        'endLine' => 97,
        'startColumn' => 5,
        'endColumn' => 39,
      ),
    ),
    'immediateProperties' => 
    array (
      'patterns' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'name' => 'patterns',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * A list of regular expression patterns used to filter the text columns of the dataset.
 *
 * @var list<string>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 104,
        'endLine' => 104,
        'startColumn' => 5,
        'endColumn' => 30,
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
          'patterns' => 
          array (
            'name' => 'patterns',
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
            'startLine' => 110,
            'endLine' => 110,
            'startColumn' => 33,
            'endColumn' => 47,
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
 * @param string[] $patterns
 * @throws InvalidArgumentException
 */',
        'startLine' => 110,
        'endLine' => 120,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'currentClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
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
        'startLine' => 129,
        'endLine' => 132,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'currentClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
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
            'startLine' => 139,
            'endLine' => 139,
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
 * @param array<mixed[]> $samples
 */',
        'startLine' => 139,
        'endLine' => 146,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'currentClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'aliasName' => NULL,
      ),
      'filter' => 
      array (
        'name' => 'filter',
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
            'byRef' => true,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 153,
            'endLine' => 153,
            'startColumn' => 31,
            'endColumn' => 44,
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
 * Filter the regex patterns from the dataset.
 *
 * @param list<mixed> $sample
 */',
        'startLine' => 153,
        'endLine' => 160,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'currentClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
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
        'startLine' => 169,
        'endLine' => 172,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Transformers',
        'declaringClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'implementingClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
        'currentClassName' => 'Rubix\\ML\\Transformers\\RegexFilter',
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