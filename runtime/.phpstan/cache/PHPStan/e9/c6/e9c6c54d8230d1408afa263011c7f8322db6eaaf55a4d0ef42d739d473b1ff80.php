<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionFunction-substr
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'name' => 'substr',
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
        'startLine' => 77,
        'endLine' => 77,
        'startColumn' => 21,
        'endColumn' => 34,
        'parameterIndex' => 0,
        'isOptional' => false,
      ),
      'offset' => 
      array (
        'name' => 'offset',
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
        'startLine' => 77,
        'endLine' => 77,
        'startColumn' => 37,
        'endColumn' => 47,
        'parameterIndex' => 1,
        'isOptional' => false,
      ),
      'length' => 
      array (
        'name' => 'length',
        'default' => 
        array (
          'code' => '\\null',
          'attributes' => 
          array (
            'startLine' => 77,
            'endLine' => 77,
            'startTokenPos' => 52,
            'startFilePos' => 3200,
            'endTokenPos' => 52,
            'endFilePos' => 3203,
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
        'startLine' => 77,
        'endLine' => 77,
        'startColumn' => 50,
        'endColumn' => 68,
        'parameterIndex' => 2,
        'isOptional' => true,
      ),
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
      0 => 
      array (
        'name' => 'JetBrains\\PhpStorm\\Pure',
        'isRepeated' => false,
        'arguments' => 
        array (
        ),
      ),
      1 => 
      array (
        'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
        'isRepeated' => false,
        'arguments' => 
        array (
          0 => 
          array (
            'code' => '["8.0" => "string"]',
            'attributes' => 
            array (
              'startLine' => 76,
              'endLine' => 76,
              'startTokenPos' => 15,
              'startFilePos' => 3089,
              'endTokenPos' => 21,
              'endFilePos' => 3107,
            ),
          ),
          'default' => 
          array (
            'code' => '"string|false"',
            'attributes' => 
            array (
              'startLine' => 76,
              'endLine' => 76,
              'startTokenPos' => 27,
              'startFilePos' => 3119,
              'endTokenPos' => 27,
              'endFilePos' => 3132,
            ),
          ),
        ),
      ),
    ),
    'docComment' => '/**
 * Returns the portion of string specified by the offset and length parameters.
 * @link https://php.net/manual/en/function.substr.php
 * @param string $string <p>
 * The input string.
 * </p>
 * @param int $offset <p>
 * If offset is non-negative, the returned string will start at the offset\'th position in string, counting from zero.
 * For instance, in the string \'abcdef\', the character at position 0 is \'a\', the character at position 2 is \'c\', and so forth.
 * </p>
 * <p>
 * If offset is negative, the returned string will start at the offset\'th character from the end of string.
 * </p>
 * <p>
 * If string is less than offset characters long, an empty string will be returned.
 * </p>
 * <p>
 * Using a negative offset
 * </p>
 * <pre>
 * <?php
 * $rest = substr("abcdef", -1);    // returns "f"
 * $rest = substr("abcdef", -2);    // returns "ef"
 * $rest = substr("abcdef", -3, 1); // returns "d"
 * ?>
 * </pre>
 * @param int|null $length [optional] <p>
 * If length is given and is positive, the string returned will contain at most length characters beginning from offset
 * (depending on the length of string).
 * </p>
 * <p>
 * If length is given and is negative, then that many characters will be omitted from the end of string.
 * If offset denotes the position of this truncation or beyond, an empty string will be returned.
 * </p>
 * <p>
 * If length is given and is 0, an empty string will be returned.
 * </p>
 * <p>
 * Starting from PHP 8.0 if length is omitted or null, the substring starting from offset until the end of the string will be returned.
 * </p>
 * <p>
 * Using a negative length:
 * </p>
 * <pre>
 * <?php
 * $rest = substr("abcdef", 0, -1);  // returns "abcde"
 * $rest = substr("abcdef", 2, -1);  // returns "cde"
 * $rest = substr("abcdef", 4, -4);  // returns false
 * $rest = substr("abcdef", -3, -1); // returns "de"
 * ?>
 * </pre>
 * @return string|false Returns the extracted part of string, or an empty string. (FALSE prior PHP 8.0)
 *  <p>
 *   Basic usage:
 *  </p>
 *   <code>
 *   echo substr(\'abcdef\', 1), PHP_EOL;     // bcdef
 *   echo substr("abcdef", 1, null), PHP_EOL; // bcdef; prior to PHP 8.0.0, empty string was returned
 *   echo substr(\'abcdef\', 1, 3), PHP_EOL;  // bcd
 *   echo substr(\'abcdef\', 0, 4), PHP_EOL;  // abcd
 *   echo substr(\'abcdef\', 0, 8), PHP_EOL;  // abcdef
 *   echo substr(\'abcdef\', -1, 1), PHP_EOL; // f
 *
 *   // Accessing single characters in a string
 *   // can also be achieved using "square brackets"
 *   $string = \'abcdef\';
 *   echo $string[0], PHP_EOL;                 // a
 *   echo $string[3], PHP_EOL;                 // d
 *   echo $string[strlen($string)-1], PHP_EOL; // f
 *   </code>
 */',
    'startLine' => 75,
    'endLine' => 79,
    'startColumn' => 5,
    'endColumn' => 5,
    'couldThrow' => false,
    'isClosure' => false,
    'isGenerator' => false,
    'isVariadic' => false,
    'isStatic' => false,
    'namespace' => NULL,
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\InternalLocatedSource',
      'data' => 
      array (
        'name' => 'substr',
        'filename' => 'phpstorm-stubs:standard/standard_1.stub',
        'extensionName' => 'standard',
        'aliasName' => NULL,
      ),
    ),
  ),
));