<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionClass-splobjectstorage
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\InternalLocatedSource',
      'data' => 
      array (
        'name' => 'SplObjectStorage',
        'filename' => 'phpstorm-stubs:SPL/SPL_c1.stub',
        'extensionName' => 'SPL',
        'aliasName' => NULL,
      ),
    ),
    'namespace' => NULL,
    'name' => 'SplObjectStorage',
    'shortName' => 'SplObjectStorage',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @template TObject of object
 * @template TValue
 * The SplObjectStorage class provides a map from objects to data or, by
 * ignoring data, an object set. This dual purpose can be useful in many
 * cases involving the need to uniquely identify objects.
 * @link https://php.net/manual/en/class.splobjectstorage.php
 * @template-implements Iterator<int, TObject>
 * @template-implements ArrayAccess<TObject, TValue>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 14,
    'endLine' => 350,
    'startColumn' => 5,
    'endColumn' => 5,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Countable',
      1 => 'SeekableIterator',
      2 => 'Serializable',
      3 => 'ArrayAccess',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      'attach' => 
      array (
        'name' => 'attach',
        'parameters' => 
        array (
          'object' => 
          array (
            'name' => 'object',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'object',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'object\']',
                    'attributes' => 
                    array (
                      'startLine' => 31,
                      'endLine' => 31,
                      'startTokenPos' => 56,
                      'startFilePos' => 1266,
                      'endTokenPos' => 62,
                      'endFilePos' => 1284,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 31,
                      'endLine' => 31,
                      'startTokenPos' => 68,
                      'startFilePos' => 1296,
                      'endTokenPos' => 68,
                      'endFilePos' => 1297,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 31,
            'endLine' => 32,
            'startColumn' => 13,
            'endColumn' => 26,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'info' => 
          array (
            'name' => 'info',
            'default' => 
            array (
              'code' => '\\null',
              'attributes' => 
              array (
                'startLine' => 34,
                'endLine' => 34,
                'startTokenPos' => 102,
                'startFilePos' => 1455,
                'endTokenPos' => 102,
                'endFilePos' => 1458,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 33,
                      'endLine' => 33,
                      'startTokenPos' => 80,
                      'startFilePos' => 1395,
                      'endTokenPos' => 86,
                      'endFilePos' => 1412,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 33,
                      'endLine' => 33,
                      'startTokenPos' => 92,
                      'startFilePos' => 1424,
                      'endTokenPos' => 92,
                      'endFilePos' => 1425,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 33,
            'endLine' => 34,
            'startColumn' => 13,
            'endColumn' => 30,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
          1 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Deprecated',
            'isRepeated' => false,
            'arguments' => 
            array (
              0 => 
              array (
                'code' => '\'use method SplObjectStorage::offset{Exists|Set|Unset}() instead\'',
                'attributes' => 
                array (
                  'startLine' => 29,
                  'endLine' => 29,
                  'startTokenPos' => 36,
                  'startFilePos' => 1086,
                  'endTokenPos' => 36,
                  'endFilePos' => 1150,
                ),
              ),
              'since' => 
              array (
                'code' => '\'8.5\'',
                'attributes' => 
                array (
                  'startLine' => 29,
                  'endLine' => 29,
                  'startTokenPos' => 42,
                  'startFilePos' => 1160,
                  'endTokenPos' => 42,
                  'endFilePos' => 1164,
                ),
              ),
            ),
          ),
        ),
        'docComment' => '/**
 * Adds an object in the storage
 * @link https://php.net/manual/en/splobjectstorage.attach.php
 * @param TObject $object <p>
 * The object to add.
 * </p>
 * @param TValue $info [optional] <p>
 * The data to associate with the object.
 * </p>
 * @return void
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 28,
        'endLine' => 37,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'detach' => 
      array (
        'name' => 'detach',
        'parameters' => 
        array (
          'object' => 
          array (
            'name' => 'object',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'object',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'object\']',
                    'attributes' => 
                    array (
                      'startLine' => 50,
                      'endLine' => 50,
                      'startTokenPos' => 142,
                      'startFilePos' => 2071,
                      'endTokenPos' => 148,
                      'endFilePos' => 2089,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 50,
                      'endLine' => 50,
                      'startTokenPos' => 154,
                      'startFilePos' => 2101,
                      'endTokenPos' => 154,
                      'endFilePos' => 2102,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 50,
            'endLine' => 51,
            'startColumn' => 13,
            'endColumn' => 26,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
          1 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Deprecated',
            'isRepeated' => false,
            'arguments' => 
            array (
              0 => 
              array (
                'code' => '\'use method SplObjectStorage::offset{Exists|Set|Unset}() instead\'',
                'attributes' => 
                array (
                  'startLine' => 48,
                  'endLine' => 48,
                  'startTokenPos' => 122,
                  'startFilePos' => 1891,
                  'endTokenPos' => 122,
                  'endFilePos' => 1955,
                ),
              ),
              'since' => 
              array (
                'code' => '\'8.5\'',
                'attributes' => 
                array (
                  'startLine' => 48,
                  'endLine' => 48,
                  'startTokenPos' => 128,
                  'startFilePos' => 1965,
                  'endTokenPos' => 128,
                  'endFilePos' => 1969,
                ),
              ),
            ),
          ),
        ),
        'docComment' => '/**
 * Removes an object from the storage
 * @link https://php.net/manual/en/splobjectstorage.detach.php
 * @param TObject $object <p>
 * The object to remove.
 * </p>
 * @return void
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 47,
        'endLine' => 54,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'contains' => 
      array (
        'name' => 'contains',
        'parameters' => 
        array (
          'object' => 
          array (
            'name' => 'object',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'object',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'object\']',
                    'attributes' => 
                    array (
                      'startLine' => 67,
                      'endLine' => 67,
                      'startTokenPos' => 200,
                      'startFilePos' => 2819,
                      'endTokenPos' => 206,
                      'endFilePos' => 2837,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 67,
                      'endLine' => 67,
                      'startTokenPos' => 212,
                      'startFilePos' => 2849,
                      'endTokenPos' => 212,
                      'endFilePos' => 2850,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 67,
            'endLine' => 68,
            'startColumn' => 13,
            'endColumn' => 26,
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
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
          1 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Deprecated',
            'isRepeated' => false,
            'arguments' => 
            array (
              0 => 
              array (
                'code' => '\'use method SplObjectStorage::offset{Exists|Set|Unset}() instead\'',
                'attributes' => 
                array (
                  'startLine' => 65,
                  'endLine' => 65,
                  'startTokenPos' => 180,
                  'startFilePos' => 2637,
                  'endTokenPos' => 180,
                  'endFilePos' => 2701,
                ),
              ),
              'since' => 
              array (
                'code' => '\'8.5\'',
                'attributes' => 
                array (
                  'startLine' => 65,
                  'endLine' => 65,
                  'startTokenPos' => 186,
                  'startFilePos' => 2711,
                  'endTokenPos' => 186,
                  'endFilePos' => 2715,
                ),
              ),
            ),
          ),
        ),
        'docComment' => '/**
 * Checks if the storage contains a specific object
 * @link https://php.net/manual/en/splobjectstorage.contains.php
 * @param TObject $object <p>
 * The object to look for.
 * </p>
 * @return bool true if the object is in the storage, false otherwise.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 64,
        'endLine' => 71,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'addAll' => 
      array (
        'name' => 'addAll',
        'parameters' => 
        array (
          'storage' => 
          array (
            'name' => 'storage',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'SplObjectStorage',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'SplObjectStorage\']',
                    'attributes' => 
                    array (
                      'startLine' => 83,
                      'endLine' => 83,
                      'startTokenPos' => 245,
                      'startFilePos' => 3408,
                      'endTokenPos' => 251,
                      'endFilePos' => 3436,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 83,
                      'endLine' => 83,
                      'startTokenPos' => 257,
                      'startFilePos' => 3448,
                      'endTokenPos' => 257,
                      'endFilePos' => 3449,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 83,
            'endLine' => 84,
            'startColumn' => 13,
            'endColumn' => 37,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Adds all objects from another storage
 * @link https://php.net/manual/en/splobjectstorage.addall.php
 * @param SplObjectStorage<TObject, TValue> $storage <p>
 * The storage you want to import.
 * </p>
 * @return int
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 81,
        'endLine' => 87,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'removeAll' => 
      array (
        'name' => 'removeAll',
        'parameters' => 
        array (
          'storage' => 
          array (
            'name' => 'storage',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'SplObjectStorage',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'SplObjectStorage\']',
                    'attributes' => 
                    array (
                      'startLine' => 99,
                      'endLine' => 99,
                      'startTokenPos' => 290,
                      'startFilePos' => 4070,
                      'endTokenPos' => 296,
                      'endFilePos' => 4098,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 99,
                      'endLine' => 99,
                      'startTokenPos' => 302,
                      'startFilePos' => 4110,
                      'endTokenPos' => 302,
                      'endFilePos' => 4111,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 99,
            'endLine' => 100,
            'startColumn' => 13,
            'endColumn' => 37,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Removes objects contained in another storage from the current storage
 * @link https://php.net/manual/en/splobjectstorage.removeall.php
 * @param SplObjectStorage<TObject, TValue> $storage <p>
 * The storage containing the elements to remove.
 * </p>
 * @return int
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 97,
        'endLine' => 103,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'removeAllExcept' => 
      array (
        'name' => 'removeAllExcept',
        'parameters' => 
        array (
          'storage' => 
          array (
            'name' => 'storage',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'SplObjectStorage',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'SplObjectStorage\']',
                    'attributes' => 
                    array (
                      'startLine' => 116,
                      'endLine' => 116,
                      'startTokenPos' => 335,
                      'startFilePos' => 4810,
                      'endTokenPos' => 341,
                      'endFilePos' => 4838,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 116,
                      'endLine' => 116,
                      'startTokenPos' => 347,
                      'startFilePos' => 4850,
                      'endTokenPos' => 347,
                      'endFilePos' => 4851,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 116,
            'endLine' => 117,
            'startColumn' => 13,
            'endColumn' => 37,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Removes all objects except for those contained in another storage from the current storage
 * @link https://php.net/manual/en/splobjectstorage.removeallexcept.php
 * @param SplObjectStorage<TObject, TValue> $storage <p>
 * The storage containing the elements to retain in the current storage.
 * </p>
 * @return int
 * @since 5.3
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 114,
        'endLine' => 120,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'getInfo' => 
      array (
        'name' => 'getInfo',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'mixed',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Returns the data associated with the current iterator entry
 * @link https://php.net/manual/en/splobjectstorage.getinfo.php
 * @return TValue The data associated with the current iterator position.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 127,
        'endLine' => 130,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'setInfo' => 
      array (
        'name' => 'setInfo',
        'parameters' => 
        array (
          'info' => 
          array (
            'name' => 'info',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 142,
                      'endLine' => 142,
                      'startTokenPos' => 401,
                      'startFilePos' => 5846,
                      'endTokenPos' => 407,
                      'endFilePos' => 5863,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 142,
                      'endLine' => 142,
                      'startTokenPos' => 413,
                      'startFilePos' => 5875,
                      'endTokenPos' => 413,
                      'endFilePos' => 5876,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 142,
            'endLine' => 143,
            'startColumn' => 13,
            'endColumn' => 23,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Sets the data associated with the current iterator entry
 * @link https://php.net/manual/en/splobjectstorage.setinfo.php
 * @param TValue $info <p>
 * The data to associate with the current iterator entry.
 * </p>
 * @return void
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 140,
        'endLine' => 146,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'count' => 
      array (
        'name' => 'count',
        'parameters' => 
        array (
          'mode' => 
          array (
            'name' => 'mode',
            'default' => 
            array (
              'code' => '\\COUNT_NORMAL',
              'attributes' => 
              array (
                'startLine' => 157,
                'endLine' => 157,
                'startTokenPos' => 459,
                'startFilePos' => 6434,
                'endTokenPos' => 459,
                'endFilePos' => 6445,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\PhpStormStubsElementAvailable',
                'isRepeated' => false,
                'arguments' => 
                array (
                  'from' => 
                  array (
                    'code' => '\'8.0\'',
                    'attributes' => 
                    array (
                      'startLine' => 156,
                      'endLine' => 156,
                      'startTokenPos' => 449,
                      'startFilePos' => 6402,
                      'endTokenPos' => 449,
                      'endFilePos' => 6406,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 156,
            'endLine' => 157,
            'startColumn' => 13,
            'endColumn' => 36,
            'parameterIndex' => 0,
            'isOptional' => true,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Returns the number of objects in the storage
 * @link https://php.net/manual/en/splobjectstorage.count.php
 * @param int $mode [optional]
 * @return int The number of objects in the storage.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 154,
        'endLine' => 160,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'rewind' => 
      array (
        'name' => 'rewind',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Rewind the iterator to the first storage element
 * @link https://php.net/manual/en/splobjectstorage.rewind.php
 * @return void
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 167,
        'endLine' => 170,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'valid' => 
      array (
        'name' => 'valid',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Returns if the current iterator entry is valid
 * @link https://php.net/manual/en/splobjectstorage.valid.php
 * @return bool true if the iterator entry is valid, false otherwise.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 177,
        'endLine' => 180,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'key' => 
      array (
        'name' => 'key',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Returns the index at which the iterator currently is
 * @link https://php.net/manual/en/splobjectstorage.key.php
 * @return int The index corresponding to the position of the iterator.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 187,
        'endLine' => 190,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'current' => 
      array (
        'name' => 'current',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'object',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Returns the current storage entry
 * @link https://php.net/manual/en/splobjectstorage.current.php
 * @return TObject The object at the current iterator position.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 197,
        'endLine' => 200,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'next' => 
      array (
        'name' => 'next',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Move to the next entry
 * @link https://php.net/manual/en/splobjectstorage.next.php
 * @return void
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 207,
        'endLine' => 210,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'unserialize' => 
      array (
        'name' => 'unserialize',
        'parameters' => 
        array (
          'data' => 
          array (
            'name' => 'data',
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'string\']',
                    'attributes' => 
                    array (
                      'startLine' => 223,
                      'endLine' => 223,
                      'startTokenPos' => 591,
                      'startFilePos' => 8814,
                      'endTokenPos' => 597,
                      'endFilePos' => 8832,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 223,
                      'endLine' => 223,
                      'startTokenPos' => 603,
                      'startFilePos' => 8844,
                      'endTokenPos' => 603,
                      'endFilePos' => 8845,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 223,
            'endLine' => 224,
            'startColumn' => 13,
            'endColumn' => 24,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Unserializes a storage from its string representation
 * @link https://php.net/manual/en/splobjectstorage.unserialize.php
 * @param string $data <p>
 * The serialized representation of a storage.
 * </p>
 * @return void
 * @since 5.2
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 221,
        'endLine' => 227,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'serialize' => 
      array (
        'name' => 'serialize',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Serializes the storage
 * @link https://php.net/manual/en/splobjectstorage.serialize.php
 * @return string A string representing the storage.
 * @since 5.2
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 235,
        'endLine' => 238,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'offsetExists' => 
      array (
        'name' => 'offsetExists',
        'parameters' => 
        array (
          'object' => 
          array (
            'name' => 'object',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 250,
            'endLine' => 250,
            'startColumn' => 38,
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
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Checks whether an object exists in the storage
 * @link https://php.net/manual/en/splobjectstorage.offsetexists.php
 * @param TObject $object <p>
 * The object to look for.
 * </p>
 * @return bool true if the object exists in the storage,
 * and false otherwise.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 249,
        'endLine' => 252,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'offsetSet' => 
      array (
        'name' => 'offsetSet',
        'parameters' => 
        array (
          'object' => 
          array (
            'name' => 'object',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.1\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 267,
                      'endLine' => 267,
                      'startTokenPos' => 679,
                      'startFilePos' => 10403,
                      'endTokenPos' => 685,
                      'endFilePos' => 10420,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 267,
                      'endLine' => 267,
                      'startTokenPos' => 691,
                      'startFilePos' => 10432,
                      'endTokenPos' => 691,
                      'endFilePos' => 10433,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 267,
            'endLine' => 268,
            'startColumn' => 13,
            'endColumn' => 25,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'info' => 
          array (
            'name' => 'info',
            'default' => 
            array (
              'code' => '\\null',
              'attributes' => 
              array (
                'startLine' => 270,
                'endLine' => 270,
                'startTokenPos' => 725,
                'startFilePos' => 10590,
                'endTokenPos' => 725,
                'endFilePos' => 10593,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 269,
                      'endLine' => 269,
                      'startTokenPos' => 703,
                      'startFilePos' => 10530,
                      'endTokenPos' => 709,
                      'endFilePos' => 10547,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 269,
                      'endLine' => 269,
                      'startTokenPos' => 715,
                      'startFilePos' => 10559,
                      'endTokenPos' => 715,
                      'endFilePos' => 10560,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 269,
            'endLine' => 270,
            'startColumn' => 13,
            'endColumn' => 30,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Associates data to an object in the storage
 * @link https://php.net/manual/en/splobjectstorage.offsetset.php
 * @param TObject $object <p>
 * The object to associate data with.
 * </p>
 * @param TValue $info [optional] <p>
 * The data to associate with the object.
 * </p>
 * @return void
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 265,
        'endLine' => 273,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'offsetUnset' => 
      array (
        'name' => 'offsetUnset',
        'parameters' => 
        array (
          'object' => 
          array (
            'name' => 'object',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 284,
            'endLine' => 284,
            'startColumn' => 37,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Removes an object from the storage
 * @link https://php.net/manual/en/splobjectstorage.offsetunset.php
 * @param TObject $object <p>
 * The object to remove.
 * </p>
 * @return void
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 283,
        'endLine' => 286,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'offsetGet' => 
      array (
        'name' => 'offsetGet',
        'parameters' => 
        array (
          'object' => 
          array (
            'name' => 'object',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 297,
            'endLine' => 297,
            'startColumn' => 35,
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
            'name' => 'mixed',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Returns the data associated with an <type>object</type>
 * @link https://php.net/manual/en/splobjectstorage.offsetget.php
 * @param TObject $object <p>
 * The object to look for.
 * </p>
 * @return TValue The data previously associated with the object in the storage.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 296,
        'endLine' => 299,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'getHash' => 
      array (
        'name' => 'getHash',
        'parameters' => 
        array (
          'object' => 
          array (
            'name' => 'object',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'object',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'object\']',
                    'attributes' => 
                    array (
                      'startLine' => 312,
                      'endLine' => 312,
                      'startTokenPos' => 796,
                      'startFilePos' => 12139,
                      'endTokenPos' => 802,
                      'endFilePos' => 12157,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 312,
                      'endLine' => 312,
                      'startTokenPos' => 808,
                      'startFilePos' => 12169,
                      'endTokenPos' => 808,
                      'endFilePos' => 12170,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 312,
            'endLine' => 313,
            'startColumn' => 13,
            'endColumn' => 26,
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
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Calculate a unique identifier for the contained objects
 * @link https://php.net/manual/en/splobjectstorage.gethash.php
 * @param TObject $object <p>
 * object whose identifier is to be calculated.
 * </p>
 * @return string A string with the calculated identifier.
 * @since 5.4
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 310,
        'endLine' => 316,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      '__serialize' => 
      array (
        'name' => '__serialize',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * @return array
 * @since 7.4
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 322,
        'endLine' => 325,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      '__unserialize' => 
      array (
        'name' => '__unserialize',
        'parameters' => 
        array (
          'data' => 
          array (
            'name' => 'data',
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
            'startLine' => 332,
            'endLine' => 332,
            'startColumn' => 39,
            'endColumn' => 49,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * @param array $data
 * @since 7.4
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 331,
        'endLine' => 334,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      '__debugInfo' => 
      array (
        'name' => '__debugInfo',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * @return array
 * @since 7.4
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 340,
        'endLine' => 343,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
        'aliasName' => NULL,
      ),
      'seek' => 
      array (
        'name' => 'seek',
        'parameters' => 
        array (
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
            'startLine' => 347,
            'endLine' => 347,
            'startColumn' => 30,
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
 * @since 8.4
 */',
        'startLine' => 347,
        'endLine' => 349,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'SplObjectStorage',
        'implementingClassName' => 'SplObjectStorage',
        'currentClassName' => 'SplObjectStorage',
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