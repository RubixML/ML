<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Encoding;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_dir;
use function is_file;
use function is_readable;
use function is_writable;
use function file_get_contents;
use function file_put_contents;
use function time;

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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
     * @param \Rubix\ML\Encoding $encoding
     * @throws \RuntimeException
     */
    public function save(Encoding $encoding) : void
    {
        if (!is_file($this->path) and !is_writable(dirname($this->path))) {
            throw new RuntimeException('Folder does not exist or is not writable');
        }

        if (is_file($this->path) and !is_writable($this->path)) {
            throw new RuntimeException("File {$this->path} is not writable.");
        }

        if ($encoding->bytes() === 0) {
            throw new RuntimeException('Encoding does not contain any data.');
        }

        if ($this->history and is_file($this->path)) {
            $timestamp = (string) time();

            $filename = "{$this->path}-$timestamp." . self::HISTORY_EXT;

            $num = 0;

            while (is_file($filename)) {
                $filename = "{$this->path}-$timestamp-" . ++$num . '.' . self::HISTORY_EXT;
            }

            if (!rename($this->path, $filename)) {
                throw new RuntimeException('Could not create history file.');
            }
        }

        $success = file_put_contents($this->path, $encoding->data(), LOCK_EX);

        if (!$success) {
            throw new RuntimeException('Could not write to the filesystem.');
        }
    }

    /**
     * Load a persisted encoding.
     *
     * @throws \RuntimeException
     * @return \Rubix\ML\Encoding
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
