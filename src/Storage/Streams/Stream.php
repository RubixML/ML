<?php

namespace Rubix\ML\Storage\Streams;

use Generator;
use IteratorAggregate;
use Stringable;

use const PHP_EOL;
use const SEEK_SET;

/**
 * Stream.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 *
 * @extends \IteratorAggregate<string>
 */
interface Stream extends IteratorAggregate, Stringable
{
    /**
     * @var string
     */
    public const READ_ONLY = 'r';

    /**
     * @var string
     */
    public const READ_WRITE = 'r+';

    /**
     * @var string
     */
    public const WRITE_ONLY = 'w';

    /**
     * @return bool
     */
    public function open() : bool;

    /**
     * @return array<mixed>
     */
    public function metadata() : array;

    /**
     * @param string $key
     * @param mixed $fallback
     * @return mixed
     */
    public function meta(string $key, $fallback = '');

    /**
     * @return string
     */
    public function mode() : string;

    /**
     * @return bool
     */
    public function readable() : bool;

    /**
     * @return bool
     */
    public function writable() : bool;

    /**
     * @return bool
     */
    public function seekable() : bool;

    /**
     * Read one line from the stream.
     *
     * @param ?int $length
     * @param string $ending
     * @return string
     */
    public function line(?int $length = null, string $ending = PHP_EOL);

    /**
     * Read bytes from the stream.
     *
     * @param ?int $length
     * @return string
     */
    public function read(?int $length = null) : string;

    /**
     * Read the remaining data from the stream until EOF.
     *
     * @throws \Rubix\ML\Storage\Exceptions\RuntimeException
     * @return string
     */
    public function contents() : string;

    /**
     * Check for end-of-file on the internal file pointer
     *
     * @see https://www.php.net/manual/en/function.feof.php
     *
     * @return bool
     */
    public function eof() : bool;

    /**
     * Write data to the stream. Return the number of bytes written
     *
     * @see https://www.php.net/manual/en/function.fwrite.php
     *
     * @param string $data
     *
     * @throws \Rubix\ML\Storage\Exceptions\RuntimeException
     * @return int
     */
    public function write(string $data) : int;

    /**
     * Get the position of the file pointer.
     *
     * @see https://www.php.net/manual/en/function.ftell.php
     *
     * @throws \Rubix\ML\Storage\Exceptions\RuntimeException
     * @return int
     */
    public function tell() : int;

    /**
     * Move the file pointer to a new position
     *
     * @see https://php.net/manual/en/function.fseek.php
     *
     * @param int $offset
     * @param int $whence
     *
     * @throws \Rubix\ML\Storage\Exceptions\RuntimeException
     */
    public function seek(int $offset, int $whence = SEEK_SET) : void;

    /**
     * Move the file pointer to the beginning of the stream.
     *
     * @see https://php.net/manual/en/function.rewind.php
     */
    public function rewind() : void;

    /**
     * Close the pointer to the stream.
     *
     * @see https://php.net/manual/en/function.fclose.php
     */
    public function close() : void;

    /**
     * @return Generator<string>
     */
    public function getIterator() : Generator;
}
