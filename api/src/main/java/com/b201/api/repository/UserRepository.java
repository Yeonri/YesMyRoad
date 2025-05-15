package com.b201.api.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import com.b201.api.domain.User;

@Repository
public interface UserRepository extends JpaRepository<User, Integer> {

	@Query("""
		select u from User u join fetch u.region where u.username = :username
		""")
	Optional<User> findByUsername(String username);
}
