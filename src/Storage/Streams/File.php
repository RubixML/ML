<?php

namespace Rubix\ML\Storage\Streams;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Storage\Exceptions\WriteError;
use Rubix\ML\Storage\Exceptions\RuntimeException;
use Generator;

/**
 * Stream.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
class File implements Stream
{
    /**
     * @var int
     */
    protected const DEFAULT_BUFFER = 1048576;

    /**
     * @var array
     */
    protected const MODES = [
        'readable' => [
            'a+' => true, 'a+b' => true, 'a+t' => true,
            'c+' => true, 'c+b' => true, 'c+t' => true,
            'r' => true, 'r+' => true, 'r+b' => true,
            'r+t' => true, 'rb' => true, 'rt' => true,
            'w+' => true, 'w+b' => true, 'w+t' => true,
            'x+' => true, 'x+b' => true, 'x+t' => true,
        ],
        'writable' => [
            'a' => true, 'a+' => true, 'a+b' => true,
            'a+t' => true, 'ab' => true, 'at' => true,
            'c' => true, 'c+' => true, 'c+b' => true,
            'c+t' => true, 'cb' => true, 'ct' => true,
            'r+' => true, 'r+b' => true, 'r+t' => true,
            'w' => true, 'w+' => true, 'w+b' => true,
            'w+t' => true, 'wb' => true, 'wt' => true,
            'x' => true, 'x+' => true, 'x+b' => true,
            'x+t' => true, 'xb' => true, 'xt' => true,
        ],
    ];

    /**
     * @var resource
     */
    protected $stream;

    /**
     * @var int
     */
    protected $buffer;

    /**
     * @var bool
     */
    protected $open;

    /**
     * @param mixed $input
     * @param string $mode
     * @param int $buffer
     * @throws \InvalidArgumentException
     * @return \Rubix\ML\Storage\Streams\Stream
     */
    public static function factory($input, string $mode = Stream::READ_WRITE, int $buffer = self::DEFAULT_BUFFER) : Stream
    {
        $type = gettype($input);

        if ($type == 'string' or ($type === 'object' and method_exists($input, '__toString'))) {
            $resource = self::resource((string) $input, $mode);

            return new self($resource, $buffer);
        }

        throw new InvalidArgumentException("Could not make stream from type: $type");
    }

    /**
     * @param string $contents
     * @param string $mode
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @throws \Rubix\ML\Storage\Exceptions\RuntimeException
     * @return resource
     */
    public static function resource(string $contents, string $mode = Stream::READ_WRITE)
    {
        if (!array_key_exists($mode, self::MODES['readable']) and !array_key_exists($mode, self::MODES['writable'])) {
            throw new InvalidArgumentException("Unsupported mode: {$mode}");
        }

        $stream = fopen('php://temp', $mode);

        if (!$stream) {
            throw new RuntimeException('Could not create stream buffer');
        }

        fwrite($stream, $contents);
        rewind($stream);

        return $stream;
    }

    /**
     * @param resource|mixed $stream
     * @param int $buffer
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct($stream, int $buffer = self::DEFAULT_BUFFER)
    {
        if (!is_resource($stream)) {
            throw new InvalidArgumentException('Stream requires a resource');
        }

        $this->stream = $stream;
        $this->buffer = $buffer;
        $this->open = true;
    }

    /**
     * @return bool
     */
    public function open() : bool
    {
        return $this->open;
    }

    /**
     * @return array<mixed>
     */
    public function metadata() : array
    {
        return stream_get_meta_data($this->stream);
    }

    /**
     * @param string $key
     * @param mixed $fallback
     *
     * @return mixed
     */
    public function meta(string $key, $fallback = '')
    {
        switch ($key) {
            case 'readable':
                return array_key_exists($this->mode(), self::MODES['readable']);

            case 'writable':
                return array_key_exists($this->mode(), self::MODES['writable']);

            default:
                return ($this->metadata()[$key] ?? $fallback) ?: $fallback;
        }
    }

    /**
     * @return string
     */
    public function mode() : string
    {
        return strtolower((string) $this->meta('mode'));
    }

    /**
     * @return bool
     */
    public function readable() : bool
    {
        return (bool) $this->meta('readable');
    }

    /**
     * @return bool
     */
    public function writable() : bool
    {
        return (bool) $this->meta('writable');
    }

    /**
     * @return bool
     */
    public function seekable() : bool
    {
        return (bool) $this->meta('seekable');
    }

    /**
     * Read one line from the stream.
     *
     * @param ?int $length
     * @param string $ending
     *
     * @return string
     */
    public function line(?int $length = null, string $ending = PHP_EOL)
    {
        $this->assertReadable();

        return (string) stream_get_line($this->stream, $length ?? $this->buffer, $ending);
    }

    /**
     * Read bytes from the stream.
     *
     * @param ?int $length
     * @return string
     */
    public function read(?int $length = null) : string
    {
        $this->assertReadable();

        $bytes = fread($this->stream, $length ?? $this->buffer);

        if ($bytes === false) {
            throw new RuntimeException('Cannot read stream');
        }

        return $bytes;
    }

    /**
     * Read the remaining data from the stream until EOF.
     *
     * @throws \Rubix\ML\Storage\Exceptions\RuntimeException
     * @return string
     */
    public function contents() : string
    {
        $this->assertReadable();
        $contents = stream_get_contents($this->stream);

        if (false === $contents) {
            throw new RuntimeException('Could not get contents of stream');
        }

        return $contents;
    }

    /**
     * Check for end-of-file on the internal file pointer
     *
     * @see https://www.php.net/manual/en/function.feof.php
     *
     * @return bool
     */
    public function eof() : bool
    {
        return feof($this->stream);
    }

    /**
     * Write data to the stream.
     *
     * @see https://www.php.net/manual/en/function.fwrite.php
     *
     * @param string $data The string that is to be written.
     * @throws \Rubix\ML\Storage\Exceptions\RuntimeException
     * @return int Number of bytes written
     */
    public function write(string $data) : int
    {
        $this->assertWritable();
        $bytes = fwrite($this->stream, $data);

        if (false === $bytes) {
            throw new RuntimeException('Could not write to stream');
        }

        return $bytes;
    }

    /**
     * Get the position of the file pointer.
     *
     * @see https://www.php.net/manual/en/function.ftell.php
     *
     * @throws \Rubix\ML\Storage\Exceptions\RuntimeException
     * @return int
     */
    public function tell() : int
    {
        $this->assertSeekable();
        $offset = ftell($this->stream);

        if (false === $offset) {
            throw new RuntimeException('Could not get offset of stream');
        }

        return $offset;
    }

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
    public function seek(int $offset, int $whence = SEEK_SET) : void
    {
        $this->assertSeekable();

        if (0 !== fseek($this->stream, $offset, $whence)) {
            throw new RuntimeException('Could not seek on stream');
        }
    }

    /**
     * Move the file pointer to the beginning of the stream.
     *
     * @see https://php.net/manual/en/function.rewind.php
     */
    public function rewind() : void
    {
        $this->assertSeekable();

        if (false === rewind($this->stream)) {
            throw new RuntimeException('Could not rewind stream');
        }
    }

    /**
     * Close the pointer to the stream.
     *
     * @see https://php.net/manual/en/function.fclose.php
     */
    public function close() : void
    {
        if (!$this->open) {
            throw new WriteError('Cannot close stream: already closed');
        }

        if (!fclose($this->stream)) {
            throw new RuntimeException('Could not close stream');
        }

        $this->open = false;
    }

    /**
     * @return Generator<string>
     */
    public function getIterator() : Generator
    {
        while (!$this->eof() and $this->open) {
            yield $this->line();
        }
    }

    protected function assertReadable() : void
    {
        if (!$this->open) {
            throw new WriteError('Cannot read from a closed stream');
        }

        if (!$this->readable()) {
            throw new WriteError('Cannot read from a stream with mode:' . $this->meta('mode'));
        }
    }

    protected function assertWritable() : void
    {
        if (!$this->open) {
            throw new WriteError('Cannot write to a closed stream');
        }

        if (!$this->writable()) {
            throw new WriteError('Cannot write to a stream with mode:' . $this->meta('mode'));
        }
    }

    protected function assertSeekable() : void
    {
        if (!$this->open) {
            throw new WriteError('Cannot seek on a closed stream');
        }

        if (!$this->seekable()) {
            throw new WriteError('Cannot seek on a non-seekable stream');
        }
    }

    public function __toString() : string
    {
        $params = [
            'mode' => $this->mode(),
            'open' => $this->open(),
            'readable' => $this->readable(),
            'writable' => $this->writable(),
            'seekable' => $this->seekable(),
        ];

        return 'Stream : (' . Params::stringify(array_merge($params, $this->metadata())) . ')';
    }
}
