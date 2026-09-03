<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Encoding;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_dir;
use function is_file;
use function is_readable;
use function is_writable;
use function file_get_contents;
use function time;
use function tempnam;
use function rename;
use function unlink;
use function dirname;
use function fopen;
use function fwrite;
use function fflush;
use function fsync;
use function fclose;

/**
 * Filesystem
 *
 * Filesystems are local or remote storage drives that are organized by files
 * and folders. The filesystem persister serializes models to a file at a
 * user-specified path.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Filesystem implements Persister
{
    /**
     * The prefix to give temporary files created during the save process.
     *
     * @var string
     */
    public const TEMP_PREFIX = 'rubix';

    /**
     * The extension to give files created as part of a persistable's save history.
     *
     * @var string
     */
    public const HISTORY_EXT = 'old';

    /**
     * The path to the model file on the filesystem.
     *
     * @var string
     */
    protected string $path;

    /**
     * Should we keep a history of past saves?
     *
     * @var bool
     */
    protected bool $history;

    /**
     * @param string $path
     * @param bool $history
     * @throws InvalidArgumentException
     */
    public function __construct(string $path, bool $history = false)
    {
        if (empty($path)) {
            throw new InvalidArgumentException('Path cannot be empty.');
        }

        if (is_dir($path)) {
            throw new InvalidArgumentException('Path must be to a file, folder given.');
        }

        $this->path = $path;
        $this->history = $history;
    }

    /**
     * Save an encoding.
     *
     * @param Encoding $encoding
     * @throws \RuntimeException
     */
    public function save(Encoding $encoding) : void
    {
        if ($encoding->bytes() === 0) {
            throw new RuntimeException('Encoding does not contain any data.');
        }

        $dir = dirname($this->path);

        if (!is_dir($dir) or !is_writable($dir)) {
            throw new RuntimeException('Folder does not exist or is not writable');
        }

        if (is_file($this->path) and !is_writable($this->path)) {
            throw new RuntimeException("File {$this->path} is not writable.");
        }

        if ($this->history and is_file($this->path)) {
            $timestamp = (string) time();

            $filename = "{$this->path}-{$timestamp}." . self::HISTORY_EXT;

            $num = 0;

            while (is_file($filename)) {
                $filename = "{$this->path}-$timestamp-" . ++$num . '.' . self::HISTORY_EXT;
            }

            if (!rename($this->path, $filename)) {
                throw new RuntimeException('Could not create history file.');
            }
        }

        $temp = tempnam($dir, self::TEMP_PREFIX);

        if ($temp === false) {
            throw new RuntimeException('Could not create a temporary storage file in ' . $dir . '.');
        }

        try {
            $handle = fopen($temp, 'w');

            if ($handle === false) {
                throw new RuntimeException("Could not open temp file {$temp} for writing.");
            }

            if (fwrite($handle, $encoding->data()) !== $encoding->bytes()) {
                throw new RuntimeException("Could not write all bytes to temp file {$temp}.");
            }

            if (!fflush($handle)) {
                throw new RuntimeException("Could not flush the temp file {$temp}.");
            }

            if (!fsync($handle)) {
                throw new RuntimeException("Could not sync the temp file {$temp}.");
            }

            if (!fclose($handle)) {
                throw new RuntimeException("Could not finalize the write to temp file {$temp}.");
            }

            if (!rename($temp, $this->path)) {
                throw new RuntimeException('Could not finalize the save at ' . $this->path . '.');
            }

        } finally {
            if (is_file($temp)) {
                unlink($temp);
            }
        }
    }

    /**
     * Load a persisted encoding.
     *
     * @throws \RuntimeException
     * @return Encoding
     */
    public function load() : Encoding
    {
        if (!is_file($this->path)) {
            throw new RuntimeException("File {$this->path} does not exist.");
        }

        if (!is_readable($this->path)) {
            throw new RuntimeException("File {$this->path} is not readable.");
        }

        $data = file_get_contents($this->path);

        if ($data === false) {
            throw new RuntimeException('Could not load data from filesystem.');
        }

        $encoding = new Encoding($data);

        if ($encoding->bytes() === 0) {
            throw new RuntimeException('File does not contain any data.');
        }

        return $encoding;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Filesystem (path: {$this->path}, history: " . Params::toString($this->history) . ')';
    }
}
