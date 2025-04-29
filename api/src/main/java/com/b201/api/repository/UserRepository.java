package com.b201.api.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.b201.api.domain.User;

public interface UserRepository extends JpaRepository<User, Integer> {
	
	Optional<User> findByUsername(String username);
}
