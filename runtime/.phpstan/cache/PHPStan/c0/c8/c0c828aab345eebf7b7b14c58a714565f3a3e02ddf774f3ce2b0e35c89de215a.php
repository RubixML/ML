<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionFunction-preg_split
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'name' => 'preg_split',
    'parameters' => 
    array (
      'pattern' => 
      array (
        'name' => 'pattern',
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
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 25,
        'endColumn' => 39,
        'parameterIndex' => 0,
        'isOptional' => false,
      ),
      'subject' => 
      array (
        'name' => 'subject',
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
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 42,
        'endColumn' => 56,
        'parameterIndex' => 1,
        'isOptional' => false,
      ),
      'limit' => 
      array (
        'name' => 'limit',
        'default' => 
        array (
          'code' => '-1',
          'attributes' => 
          array (
            'startLine' => 32,
            'endLine' => 32,
            'startTokenPos' => 32,
            'startFilePos' => 1254,
            'endTokenPos' => 33,
            'endFilePos' => 1255,
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
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 59,
        'endColumn' => 73,
        'parameterIndex' => 2,
        'isOptional' => true,
      ),
      'flags' => 
      array (
        'name' => 'flags',
        'default' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 32,
            'endLine' => 32,
            'startTokenPos' => 42,
            'startFilePos' => 1271,
            'endTokenPos' => 42,
            'endFilePos' => 1271,
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
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 76,
        'endColumn' => 89,
        'parameterIndex' => 3,
        'isOptional' => true,
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
              'name' => 'array',
              'isIdentifier' => true,
            ),
          ),
          1 => 
          array (
            'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
            'data' => 
            array (
              'name' => 'false',
              'isIdentifier' => true,
            ),
          ),
        ),
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
    ),
    'docComment' => '/**
 * Split string by a regular expression
 * @link https://php.net/manual/en/function.preg-split.php
 * @param string $pattern <p>
 * The pattern to search for, as a string.
 * </p>
 * @param string $subject <p>
 * The input string.
 * </p>
 * @param int $limit [optional] <p>
 * If specified, then only substrings up to <i>limit</i>
 * are returned with the rest of the string being placed in the last
 * substring. A <i>limit</i> of -1, 0 or <b>NULL</b> means "no limit"
 * and, as is standard across PHP, you can use <b>NULL</b> to skip to the
 * <i>flags</i> parameter.
 * </p>
 * @param int $flags [optional] <p>
 * <i>flags</i> can be any combination of the following
 * flags (combined with the | bitwise operator):
 * <b>PREG_SPLIT_NO_EMPTY</b>
 * If this flag is set, only non-empty pieces will be returned by
 * <b>preg_split</b>.
 * </p>
 * @return string[]|false an array containing substrings of <i>subject</i>
 * split along boundaries matched by <i>pattern</i>, or <b>FALSE</b>
 * if an error occurred.
 */',
    'startLine' => 31,
    'endLine' => 34,
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
        'name' => 'preg_split',
        'filename' => 'phpstorm-stubs:pcre/pcre.stub',
        'extensionName' => 'pcre',
        'aliasName' => NULL,
      ),
    ),
  ),
));