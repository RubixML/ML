<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;
use InvalidArgumentException;
use RuntimeException;
use Redis;

/**
 * Redis DB
 *
 * Redis is a high performance in-memory key value store that can be used to
 * persist models. The persiter requires the PHP Redis extension and a properly
 * configured Redis server.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RedisDB implements Persister
{
    /**
     * The key of the model in storage.
     *
     * @var string
     */
    protected $key;

    /**
     * The number of backups to keep.
     *
     * @var int
     */
    protected $history;

    /**
     * The connector to the Redis database.
     *
     * @var \Redis
     */
    protected $connector;

    /**
     * @param  string  $key
     * @param  int  $history
     * @param  string  $host
     * @param  int  $port
     * @param  int  $db
     * @param  string  $password
     * @param  float  $timeout
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function __construct(string $key, int $history = 2, string $host = '127.0.0.1', int $port = 6379,
                                int $db = 0, string $password = null, float $timeout = 2.5)
    {
        if (!extension_loaded('redis')) {
            throw new RuntimeException('Redis extension is not loaded, check'
                . ' php.ini file.');
        }

        if ($history < 0) {
            throw new InvalidArgumentException("The number of backups to keep"
                . " cannot be less than 0, $history given.");
        }

        $connector = new Redis();

        if (!$connector->connect($host, $port, $timeout)) {
            throw new RuntimeException('Could not connect to Redis server'
                . ' at host ' . $host . ' on port ' . (string) $port . '.');
        };

        if (isset($password)) {
            if (!$connector->auth($password)) {
                throw new RuntimeException('Password is invalid.');
            }
        }

        if (!$connector->select($db)) {
            throw new RuntimeException('Could not select database number'
                . (string) $db . '.');
        };

        $this->key = $key;
        $this->history = $history;
        $this->connector = $connector;
    }

    /**
     * Return an associative array of info from the Redis server.
     *
     * @return array
     */
    public function info() : array
    {
        return $this->connector->info();
    }

    /**
     * Save the persitable object.
     *
     * @param  \Rubix\ML\Persistable  $persistable
     * @throws \RuntimeException
     * @return void
     */
    public function save(Persistable $persistable) : void
    {
        $data = serialize($persistable);

        $success = $length = $this->connector->rPush($this->key, $data);

        if (!$success) {
            throw new RuntimeException('There was an error saving the'
                . ' model to the database.');
        };

        $diff = $length - ($this->history + 1);

        if ($diff > 0) {
            for ($i = 0; $i < $diff; $i++) {
                $success = $this->connector->lPop($this->key);

                if (!$success) {
                    throw new RuntimeException('There was an error'
                        . ' deleting a backup from the database.');
                };
            }
        }
    }

    /**
     * Load a model given a version number where 0 is the last model saved.
     * 
     * @param  int  $version
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function load(int $version = 0) : Persistable
    {
        if ($version < 0) {
            throw new InvalidArgumentException("Version cannot be less"
                . " than 0, $version given.");
        }

        if ($version > $this->history) {
            throw new InvalidArgumentException("The maximum number of"
                . " backups is $this->history, $version given.");
        }

        $index = -($version + 1);

        $data = $this->connector->lGet($this->key, $index) ?: '';

        if (empty($data)) {
            throw new RuntimeException('Model does not exist in database.');
        }

        $persistable = unserialize($data);

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Model could not be reconstituted.');
        }

        return $persistable;
    }
}
