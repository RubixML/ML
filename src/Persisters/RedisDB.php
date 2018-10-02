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
     * The key of the object in storage.
     *
     * @var string
     */
    protected $key;

    /**
     * The connector to the Redis database.
     *
     * @var \Redis
     */
    protected $connector;

    /**
     * @param  string  $key
     * @param  string  $host
     * @param  int  $port
     * @param  int  $db
     * @param  string  $password
     * @param  float  $timeout
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function __construct(string $key, string $host = '127.0.0.1', int $port = 6379,
                                int $db = 0, string $password = null, float $timeout = 2.5)
    {
        if (!extension_loaded('redis')) {
            throw new RuntimeException('Redis extension is not loaded. Check'
                . ' php.ini file.');
        }

        $connector = new Redis();

        if ($connector->connect($host, $port, $timeout) === false) {
            throw new RuntimeException('Could not connect to Redis server'
                . ' at host ' . $host . ' on port ' . (string) $port . '.');
        };

        if (isset($password)) {
            if ($connector->auth($password) === false) {
                throw new InvalidArgumentException('Password is invalid.');
            }
        }

        if ($connector->select($db) === false) {
            throw new RuntimeException('Could not select database number'
                . (string) $db . '.');
        };

        $this->key = $key;
        $this->connector = $connector;
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

        if ($this->connector->set($this->key, $data) === false) {
            throw new RuntimeException('There was an error saving the object.');
        };
    }

    /**
     * Restore the persistable object.
     *
     * @throws \RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable
    {
        $data = $this->connector->get($this->key) ?: '';

        if (empty($data)) {
            throw new RuntimeException('Object does not exist in database.');
        }

        $persistable = unserialize($data);

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Object cannot be reconstituted.');
        }

        return $persistable;
    }

    /**
     * Remove the key from the database.
     *
     * @return void
     */
    public function delete() : void
    {
        $this->connector->delete($this->key);
    }
}
