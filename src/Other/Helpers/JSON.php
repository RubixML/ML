<?php

namespace Rubix\ML\Other\Helpers;

use Rubix\ML\Exceptions\RuntimeException;

use const JSON_ERROR_NONE;
use const JSON_ERROR_SYNTAX;
use const JSON_ERROR_DEPTH;
use const JSON_ERROR_STATE_MISMATCH;
use const JSON_ERROR_CTRL_CHAR;
use const JSON_ERROR_UTF8;
use const JSON_ERROR_UTF16;
use const JSON_ERROR_RECURSION;
use const JSON_ERROR_UNSUPPORTED_TYPE;
use const JSON_ERROR_INVALID_PROPERTY_NAME;
use const JSON_ERROR_INF_OR_NAN;

/**
 * JSON
 *
 * A helper class providing functions relating to JSON (Javascript Object Notation).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
class JSON
{
    /**
     * The default maximum stack depth.
     *
     * @var int
     */
    protected const DEFAULT_DEPTH = 512;

    /**
     * @var int
     */
    protected const DEFAULT_OPTIONS = 0;

    /**
     * Deserialize JSON.
     *
     * @param string $json
     * @param int $options
     * @param int $depth
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return array<mixed>
     */
    public static function decode(string $json, int $options = self::DEFAULT_OPTIONS, int $depth = self::DEFAULT_DEPTH) : array
    {
        $value = json_decode($json, true, $depth, $options);

        $error = json_last_error();

        switch ($error) {
            case JSON_ERROR_NONE:
                return $value;

            case JSON_ERROR_SYNTAX:
                throw new RuntimeException('Syntax error.');

            case JSON_ERROR_DEPTH:
                throw new RuntimeException("Maximum stack depth of $depth exceeded.");

            case JSON_ERROR_STATE_MISMATCH:
                throw new RuntimeException('Invalid or malformed JSON.');

            case JSON_ERROR_CTRL_CHAR:
                throw new RuntimeException('Unexpected control character found.');

            case JSON_ERROR_RECURSION:
                throw new RuntimeException('Recursive references detected.');

            case JSON_ERROR_UTF8:
                throw new RuntimeException('Malformed UTF-8 characters, check encoding.');

            case JSON_ERROR_UTF16:
                throw new RuntimeException('Malformed UTF-16 characters, check encoding.');

            case JSON_ERROR_UNSUPPORTED_TYPE:
                throw new RuntimeException('Unsupported type encountered.');

            case JSON_ERROR_INVALID_PROPERTY_NAME:
                throw new RuntimeException('Invalid property name encountered.');

            case JSON_ERROR_INF_OR_NAN:
                throw new RuntimeException('INF or NAN values values cannot be encoded.');

            default:
                throw new RuntimeException("Unknown JSON error. (code: $error')");
        }
    }

    /**
     * Serialize JSON.
     *
     * @param mixed $value
     * @param int $options
     * @param int $depth
     *
     * @throws \RuntimeException
     *
     * @return string
     */
    public static function encode($value, int $options = self::DEFAULT_OPTIONS, int $depth = self::DEFAULT_DEPTH) : string
    {
        $data = json_encode($value, $options, $depth);

        if ($data === false) {
            throw new RuntimeException('Could not json_encode provided data.');
        }

        return $data;
    }
}
