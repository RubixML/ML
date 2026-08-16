<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Helpers/Graphviz.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Helpers\Graphviz
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-0bbbdc4d69667d74d83277bc2a0c048cf48ed4767dcfb94384464f53d26439b7',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Helpers\\Graphviz',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Helpers/Graphviz.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Helpers',
    'name' => 'Rubix\\ML\\Helpers\\Graphviz',
    'shortName' => 'Graphviz',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Graphviz
 *
 * An interface to the popular Graphviz program for generating graph images.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 23,
    'endLine' => 76,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
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
    ),
    'immediateMethods' => 
    array (
      'dotToImage' => 
      array (
        'name' => 'dotToImage',
        'parameters' => 
        array (
          'dot' => 
          array (
            'name' => 'dot',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Encoding',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 37,
            'endLine' => 37,
            'startColumn' => 39,
            'endColumn' => 51,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'format' => 
          array (
            'name' => 'format',
            'default' => 
            array (
              'code' => '\'png\'',
              'attributes' => 
              array (
                'startLine' => 37,
                'endLine' => 37,
                'startTokenPos' => 88,
                'startFilePos' => 823,
                'endTokenPos' => 88,
                'endFilePos' => 827,
              ),
            ),
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
            'startLine' => 37,
            'endLine' => 37,
            'startColumn' => 54,
            'endColumn' => 75,
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
 * Produces an image from a "dot" formatted string.
 *
 * https://graphviz.org/doc/info/lang.html
 *
 * See https://graphviz.org/docs/outputs/ for supported formats
 *
 * @param Encoding $dot
 * @param string $format
 * @throws RuntimeException
 * @return Encoding
 */',
        'startLine' => 37,
        'endLine' => 75,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Helpers',
        'declaringClassName' => 'Rubix\\ML\\Helpers\\Graphviz',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\Graphviz',
        'currentClassName' => 'Rubix\\ML\\Helpers\\Graphviz',
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