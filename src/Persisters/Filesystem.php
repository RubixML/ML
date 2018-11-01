<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;
use InvalidArgumentException;
use RuntimeException;

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
    const BACKUP_EXT = '.backup';

    /**
     * The path to the model file on the filesystem.
     *
     * @var string
     */
    protected $path;

    /**
     * The number of backups to keep.
     *
     * @var int
     */
    protected $history;

    /**
     * @param  string  $path
     * @param  int  $history
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $path, int $history = 1)
    {
        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writable, check path and permissions.');
        }

        if ($history < 0) {
            throw new InvalidArgumentException("The number of backups to keep"
                . " cannot be less than 0, $history given.");
        }

        $this->path = $path;
        $this->history = $history;
    }

    /**
     * Save the persitable model.
     *
     * @param  \Rubix\ML\Persistable  $persistable
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void 
     */
    public function save(Persistable $persistable) : void
    {
        if ($this->history > 0 and is_file($this->path)) {
            $filename = $this->path . '.' . (string) time() . self::BACKUP_EXT;

            $success = rename($this->path, $filename);

            if (!$success) {
                throw new RuntimeException('Failed to save backup, check path'
                 . ' and permissions.');
            }

            $backups = [];

            foreach (glob("$this->path.*" . self::BACKUP_EXT) as $filename) {
                $backups[$filename] = filemtime($filename);
            }

            arsort($backups);

            $old = array_slice(array_keys($backups), $this->history);

            foreach ($old as $filename) {
                unlink($filename);
            }
        }

        $data = serialize($persistable);

        $success = file_put_contents($this->path, $data, LOCK_EX);

        if (!$success) {
            throw new RuntimeException('Failed to save model to the'
                . ' filesystem, check path and permissions.');
        }
    }

    /**
     * Load the last saved model or load from backup by order of most recent.
     * 
     * @param  int  $ordinal
     * @return \Rubix\ML\Persistable
     */
    public function load(int $ordinal = 0) : Persistable
    {
        if ($ordinal < 0) {
            throw new InvalidArgumentException("Ordinal cannot be less"
                . " than 0, $ordinal given.");
        }

        if ($ordinal > $this->history) {
            throw new InvalidArgumentException("The maximum number of"
                . " backups is $this->history, $ordinal given.");
        }

        if ($ordinal === 0) {
            if (!is_file($this->path)) {
                throw new RuntimeException('File ' . basename($this->path)
                    . ' does not exist or is a folder, check the path'
                    . ' and permissions.');
            }
            
            $path = $this->path;
        } else {
            $backups = [];

            foreach (glob("$this->path.*" . self::BACKUP_EXT) as $filename) {
                $backups[$filename] = filemtime($filename);
            }

            arsort($backups);

            $backups = array_keys($backups);

            $index = $ordinal - 1;

            if (!isset($backups[$index])) {
                throw new InvalidArgumentException("Could not load model"
                    . " from storage.");
            }

            $path = $backups[$index];
        }

        $data = file_get_contents($path) ?: '';

        $persistable = unserialize($data);

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Model could not be reconstituted.');
        }

        return $persistable;
    }

    /**
     * Delete all backups from storage.
     * 
     * @return void
     */
    public function flush() : void
    {
        foreach (glob("$this->path.*" . self::BACKUP_EXT) as $filename) {
            unlink($filename);
        }
    }
}
