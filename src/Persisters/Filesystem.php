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
    const BACKUP_EXT = '.old';

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
    public function __construct(string $path, int $history = 2)
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

            if (count($backups) > $this->history) {
                arsort($backups);

                $old = array_slice(array_keys($backups), $this->history);

                foreach ($old as $filename) {
                    unlink($filename);
                }
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
     * Load the last model that was saved.
     * 
     * @throws \RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable
    {
        if (!is_file($this->path)) {
            throw new RuntimeException('File ' . basename($this->path)
                . ' does not exist or is a folder, check the path'
                . ' and permissions.');
        }
            
        $data = file_get_contents($this->path) ?: '';

        $persistable = unserialize($data);

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Model could not be reconstituted.');
        }

        return $persistable;
    }

    /**
     * Remove all backups from storage.
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
