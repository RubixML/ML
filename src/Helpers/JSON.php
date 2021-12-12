<?php

namespace Rubix\ML\Helpers;

use Rubix\ML\Exceptions\JSONException;

use function json_encode;
use function json_decode;
use function json_last_error;

use const JSON_ERROR_NONE;

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
     * Serialize to JSON.
     *
     * @param mixed $value
     * @param int $options
     * @param positive-int $depth
     * @throws \Rubix\ML\Exceptions\JSONException
     * @return string
     */
    public static function encode($value, int $options = self::DEFAULT_OPTIONS, int $depth = self::DEFAULT_DEPTH) : string
    {
        $data = json_encode($value, $options, $depth);

        if ($data === false) {
            throw new JSONException(json_last_error());
        }

        return $data;
    }

    /**
     * Deserialize from JSON.
     *
     * @param string $data
     * @param int $options
     * @param positive-int $depth
     * @throws \Rubix\ML\Exceptions\JSONException
     * @return mixed[]
     */
    public static function decode(string $data, int $options = self::DEFAULT_OPTIONS, int $depth = self::DEFAULT_DEPTH) : array
    {
        $value = json_decode($data, true, $depth, $options);

        $code = json_last_error();

        if ($code !== JSON_ERROR_NONE) {
            throw new JSONException($code);
        }

        return $value;
    }
}
